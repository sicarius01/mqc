"""Local file loading and auditable diagnostic analysis for the workbench.

No network calls. Invalid input rows stay visible; only explicitly valid rows
enter the numerical core. Cohorts absent from a baseline never borrow test data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import uuid
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import cdqc
from cdqc.geometry import mean_edge_tangent, unit_tangents
from cdqc.sampling import DS, sample_ribbon_profiles
from cdqc.evidence import gradient_of


@dataclass
class Dataset:
    image_id: str
    img: np.ndarray
    labelmap: np.ndarray | None
    records: pd.DataFrame
    px_nm: float
    metadata: dict = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)


@dataclass
class Analysis:
    datasets: list[Dataset]
    l3: pd.DataFrame
    l2: pd.DataFrame
    l1: pd.DataFrame
    category: pd.DataFrame
    stats: dict
    params: cdqc.Params
    issues: list[dict]
    baseline_ids: list[str]
    run_id: str


def validate_params(raw=None) -> cdqc.Params:
    """Validate editable settings before they reach array operations."""
    if isinstance(raw, cdqc.Params):
        raw = asdict(raw)
    raw = dict(raw or {})
    for name in ('delta_sigma_sweep', 'log_features'):
        if name in raw:
            raw[name] = tuple(raw[name])
    try:
        params = cdqc.Params(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'파라미터 이름/형식 오류: {exc}') from exc
    positive = ('win_frac', 'win_min_px', 'win_max_px', 'grad_sigma_px',
                'mad_floor_default', 'abs_scale_floor', 'z_cap')
    nonnegative = ('margin_px', 'label_inset_px', 'min_cnr_for_ratio', 'min_rise_px',
                   'rel_scale_floor', 'pooled_mad_frac')
    for name in positive + nonnegative:
        value = getattr(params, name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or (name in positive and value == 0):
            raise ValueError(f'{name}: 유한한 {"양수" if name in positive else "0 이상 수"}가 필요합니다.')
    for name, lower in (('ribbon_half_w', 0), ('noise_patch_px', 3), ('peak_suppress', 1),
                        ('local_window', 3), ('min_seq_len', 1), ('min_cohort_n', 1)):
        value = getattr(params, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < lower:
            raise ValueError(f'{name}: {lower} 이상 정수가 필요합니다.')
    if params.local_window % 2 != 1:
        raise ValueError('local_window는 홀수여야 합니다.')
    if params.win_min_px > params.win_max_px:
        raise ValueError('win_min_px는 win_max_px 이하여야 합니다.')
    if not 0 <= params.trim_frac < .5 or not 0 < params.npk_ratio <= 1:
        raise ValueError('trim_frac은 [0, 0.5), npk_ratio는 (0, 1] 범위여야 합니다.')
    if params.method not in ('robust_linear', 'hampel'):
        raise ValueError('method는 robust_linear 또는 hampel이어야 합니다.')
    if not params.delta_sigma_sweep or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in params.delta_sigma_sweep):
        raise ValueError('delta_sigma_sweep은 유한한 양수 목록이어야 합니다.')
    for name, value in params.mad_floors.items():
        if name not in cdqc.BY_NAME or not np.isfinite(value) or value <= 0:
            raise ValueError(f'mad_floors 설정 오류: {name}')
    if any(name not in cdqc.BY_NAME for name in params.log_features):
        raise ValueError('log_features에 알 수 없는 피쳐가 있습니다.')
    if any(name not in cdqc.BY_NAME or value not in ('low', 'high', 'both') for name, value in params.direction_overrides.items()):
        raise ValueError('direction_overrides의 피쳐/방향을 확인하세요.')
    return params


def _issue(code, message, image_id='', severity='warning', **extra):
    return dict(code=code, severity=severity, image_id=image_id, message=message, **extra)


def read_table(source, filename=None) -> pd.DataFrame:
    """Read UTF-8/CP949 CSV or XLSX. Bytes require an explicit filename."""
    filename = filename or (str(source) if not isinstance(source, bytes) else '')
    suffix = Path(filename).suffix.lower()
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if suffix in ('.xlsx', '.xlsm'):
        try:
            return pd.read_excel(BytesIO(raw), engine='openpyxl')
        except ImportError as exc:
            raise ValueError('XLSX 읽기에는 openpyxl이 필요합니다. GUI 의존성을 설치하세요.') from exc
    if suffix not in ('.csv', '.tsv', '.txt'):
        raise ValueError('측정 표는 CSV, TSV 또는 XLSX 형식이어야 합니다.')
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(BytesIO(raw), encoding=encoding,
                               sep='\t' if suffix == '.tsv' else ',')
        except UnicodeDecodeError:
            continue
    raise ValueError('표 인코딩을 읽지 못했습니다. UTF-8 또는 CP949 CSV를 사용하세요.')


def _read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'이미지를 읽지 못했습니다: {Path(path).name}')
    return image


def _dm_scale(path, image_shape):
    try:
        from ncempy.io.dm import dmReader
    except ImportError as exc:
        raise ValueError('DM3/DM4 읽기에는 ncempy가 필요합니다. 또는 nm/px를 직접 입력하세요.') from exc
    data = dmReader(str(path))
    shape = np.asarray(data['data']).shape
    if len(shape) != 2:
        raise ValueError(f'DM 데이터는 2차원이어야 합니다: {shape}')
    sizes = np.atleast_1d(data['pixelSize'])
    units = np.atleast_1d(data['pixelUnit'])
    if sizes.size not in (1, 2):
        raise ValueError('DM 픽셀 축 정보를 해석할 수 없습니다.')
    scales = cdqc.to_nm(sizes, units if units.size > 1 else str(units[0]))
    scales = np.repeat(scales, 2) if len(scales) == 1 else scales
    effective = scales * np.asarray(shape) / np.asarray(image_shape)
    if not np.allclose(effective[0], effective[1], rtol=1e-4):
        raise ValueError(f'비등방 픽셀/리사이즈는 지원하지 않습니다. 축별 nm/px={effective.tolist()}')
    return float(effective[0]), {'dm_shape': list(shape), 'dm_axis_nm': scales.tolist(), 'dm_dataset_index': 0,
                                  'scale_source': 'DM metadata, resized to image'}


ALIASES = {
    'sx': ['sx', 's_x', 'LineBeginX'], 'sy': ['sy', 's_y', 'LineBeginY'],
    'ex': ['ex', 'e_x', 'LineEndX'], 'ey': ['ey', 'e_y', 'LineEndY'],
    'category': ['category', 'category_id', 'MetrologyActivityName'],
    'value_nm': ['value_nm', 'measurement', 'Measurement'],
    'unit': ['unit', 'MeasurementUnit'],
}


def load_dataset(image_path, table_path, seg_path=None, dm3_path=None,
                 px_nm=None, options=None, image_id=None) -> Dataset:
    options = dict(options or {})
    iid = str(image_id or Path(image_path).stem)
    issues = []
    raw = _read_image(image_path)
    metadata = {'image_file': str(image_path), 'table_file': str(table_path),
                'dm3_file': str(dm3_path) if dm3_path else None,
                'seg_file': str(seg_path) if seg_path else None,
                'original_dtype': str(raw.dtype), 'original_shape': list(raw.shape),
                'options': options, 'scale_source': 'manual nm/px'}
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY if raw.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        issues.append(_issue('COLOR_TO_GRAY', '컬러 이미지를 OpenCV 명도 회색조로 변환했습니다.', iid))
    if raw.ndim != 2 or min(raw.shape) < 3:
        raise ValueError('최소 3×3 이상의 2D 이미지가 필요합니다.')
    if not np.isfinite(raw).all():
        raise ValueError('이미지에 NaN 또는 무한대가 있습니다.')
    metadata['original_min'] = float(raw.min())
    metadata['original_max'] = float(raw.max())
    img = cdqc.to_uint8(raw)
    if raw.dtype != np.uint8:
        issues.append(_issue('UINT8_CONVERSION', '원본을 cdqc.to_uint8로 8비트 변환했습니다. 원본 범위를 확인하세요.', iid))
    if px_nm is None:
        if not dm3_path:
            raise ValueError('nm/px 직접 입력 또는 DM3/DM4 경로가 필요합니다.')
        px_nm, dm_metadata = _dm_scale(dm3_path, img.shape)
        metadata.update(dm_metadata)
    elif dm3_path:
        issues.append(_issue('MANUAL_SCALE_OVERRIDE', '직접 입력한 nm/px를 사용합니다. DM 메타데이터는 적용하지 않습니다.', iid))
    px_nm = float(px_nm)
    if not np.isfinite(px_nm) or px_nm <= 0:
        raise ValueError('nm/px는 0보다 큰 유한한 값이어야 합니다.')

    table = read_table(table_path)
    if table.empty:
        raise ValueError('측정 표에 데이터 행이 없습니다.')
    table.columns = table.columns.map(str)
    mapping = dict(options.get('mapping') or {})
    for target, aliases in ALIASES.items():
        if target not in mapping or not mapping[target]:
            mapping[target] = next((c for c in aliases if c in table), None)
    for name in ('sx', 'sy', 'ex', 'ey', 'category'):
        if mapping.get(name) not in table:
            raise ValueError(f'{name} 컬럼을 지정하세요. 사용 가능 컬럼: {list(table.columns)}')
    rec = table.copy()
    rec['source_row'] = np.arange(len(rec)) + 2  # spreadsheet/CSV header is row 1
    for name in ('sx', 'sy', 'ex', 'ey'):
        rec[name] = pd.to_numeric(table[mapping[name]], errors='coerce')
        rec['raw_' + name] = rec[name]
    cats = table[mapping['category']]
    missing_category = cats.isna() | cats.astype(str).str.strip().eq('')
    rec['category'] = cats.where(~missing_category, '__MISSING_CATEGORY__').astype(str)
    rec['value_nm'] = np.nan
    if mapping.get('value_nm') in table:
        values = pd.to_numeric(table[mapping['value_nm']], errors='coerce').to_numpy(float)
        unit_col = mapping.get('unit')
        unit = options.get('value_unit') or (table[unit_col].to_numpy() if unit_col in table else 'nm')
        rec['value_nm'] = cdqc.to_nm(values, unit)
        if unit_col not in table and not options.get('value_unit') and mapping['value_nm'] != 'value_nm':
            issues.append(_issue('ASSUMED_VALUE_NM', '측정값 단위 컬럼이 없어 nm로 해석했습니다. 단위 설정을 확인하세요.', iid))

    mode = options.get('coordinate_mode', 'auto')
    if mode == 'auto':
        mode = 'center_m' if mapping['sx'] == 'LineBeginX' else 'pixel'
    if mode not in ('pixel', 'center_m'):
        raise ValueError(f'알 수 없는 coordinate_mode: {mode}')
    origin = options.get('origin', 'center_half' if mode == 'center_m' else 'zero')
    if origin not in ('zero', 'one', 'center_half', 'center_pixel'):
        raise ValueError(f'알 수 없는 origin: {origin}')
    flip = bool(options.get('y_flip', mode == 'center_m'))
    h, w = img.shape
    for prefix in ('s', 'e'):
        coords = rec[[prefix + 'x', prefix + 'y']].to_numpy(float)
        if options.get('swap_xy', False):
            coords = coords[:, ::-1].copy()
        if mode == 'center_m':
            coords /= px_nm * 1e-9
            if flip:
                coords[:, 1] *= -1
            center = np.array([w, h], float) / 2
            if origin == 'center_pixel':
                center -= .5
            coords += center
        else:
            if origin == 'one':
                coords -= 1
            if flip:
                coords[:, 1] = h - 1 - coords[:, 1]
        coords += [float(options.get('offset_x', 0)), float(options.get('offset_y', 0))]
        rec[[prefix + 'x', prefix + 'y']] = coords
    metadata.update(mapping=mapping, coordinate_mode=mode, origin=origin,
                    y_flip=flip, width=w, height=h, px_nm=px_nm)
    xyz = rec[['sx', 'sy', 'ex', 'ey']].to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    inside_s = (xyz[:, 0] >= 0) & (xyz[:, 0] <= w-1) & (xyz[:, 1] >= 0) & (xyz[:, 1] <= h-1)
    inside_e = (xyz[:, 2] >= 0) & (xyz[:, 2] <= w-1) & (xyz[:, 3] >= 0) & (xyz[:, 3] <= h-1)
    zero = np.linalg.norm(xyz[:, 2:] - xyz[:, :2], axis=1) < 1e-9
    rec['inside_s'], rec['inside_e'] = inside_s, inside_e
    rec['analysis_valid'] = finite & inside_s & inside_e & ~zero & ~missing_category.to_numpy()
    for bad, code, message in ((~finite, 'NONFINITE_COORD', '유한하지 않은 좌표'),
            (finite & ~inside_s, 'S_OUTSIDE', 'S 좌표 이미지 밖'),
            (finite & ~inside_e, 'E_OUTSIDE', 'E 좌표 이미지 밖'),
            (zero, 'ZERO_LENGTH', '길이가 0인 CD'),
            (missing_category.to_numpy(), 'MISSING_CATEGORY', '카테고리 결측')):
        if bad.any():
            issues.append(_issue(code, message + ': 행을 보존하고 계산에서 제외합니다.', iid,
                                 count=int(bad.sum()), source_rows=rec.loc[bad, 'source_row'].tolist()))
    order_col = options.get('order_col')
    if order_col:
        if order_col not in rec:
            raise ValueError(f'순서 컬럼이 없습니다: {order_col}')
        rec = rec.sort_values(['category', order_col], kind='stable')
    rec['cd_index'] = rec.groupby('category', sort=False).cumcount()
    rec = rec.reset_index(drop=True)
    labelmap = None
    if seg_path:
        seg = _read_image(seg_path)
        metadata['seg_file'] = str(seg_path)
        if seg.ndim == 3:
            if options.get('seg_restore', False):
                seg = cdqc.close_annotation(seg[:, :, :3])[..., 1]
                issues.append(_issue('SEG_RESTORED', '컬러 주석 복구 후 G 채널을 라벨로 사용했습니다.', iid))
            else:
                _, labels = np.unique(seg.reshape(-1, seg.shape[2]), axis=0, return_inverse=True)
                seg = labels.reshape(seg.shape[:2]).astype(np.int32)
                issues.append(_issue('RGB_LABELS', '서로 다른 RGB 색을 서로 다른 라벨로 해석합니다. 주석 포함 여부를 확인하세요.', iid))
        if seg.shape != img.shape:
            if not options.get('seg_resize', False):
                raise ValueError(f'세그멘테이션 크기 {seg.shape} ≠ 이미지 {img.shape}. 명시적으로 최근접 리사이즈를 선택하세요.')
            seg = cv2.resize(seg.astype(np.float64), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            issues.append(_issue('SEG_RESIZED', '세그멘테이션을 최근접 보간으로 이미지 크기에 맞췄습니다.', iid))
        labelmap = seg
    else:
        issues.append(_issue('NO_SEGMENTATION', '세그멘테이션 없음: 라벨 경계 피쳐는 계산되지 않습니다.', iid, 'info'))
    return Dataset(iid, img, labelmap, rec, px_nm, metadata, issues)


def demo_datasets() -> list[Dataset]:
    """Deterministic fixtures: three normal, one globally shifted, one local error."""
    rng = np.random.default_rng(712)
    out = []
    for index, name in enumerate(('normal_01', 'normal_02', 'normal_03', 'shift_04', 'local_05')):
        h, w = 160, 192
        lab = np.zeros((h, w), np.uint8)
        lab[:, 65:115] = 1
        img = np.clip(45 + lab.astype(float)*140 + rng.normal(0, 2, (h, w)), 0, 255).astype(np.uint8)
        y = np.linspace(22, 138, 16)
        sx, ex = np.full(16, 64.5), np.full(16, 114.5)
        sx += rng.normal(0, .12, 16); ex += rng.normal(0, .12, 16)
        if index == 3:
            sx += 5; ex += 5
        if index == 4:
            ex[7] += 9
        records = pd.DataFrame(dict(sx=sx, sy=y, ex=ex, ey=y, category='demo_CD',
            value_nm=25., source_row=np.arange(16)+2, cd_index=np.arange(16),
            analysis_valid=True, inside_s=True, inside_e=True))
        out.append(Dataset(name, img, lab, records, .5,
            {'demo': True, 'injection': 'shift +5px' if index == 3 else 'E[7] +9px' if index == 4 else 'normal',
             'coordinate_mode': 'pixel', 'origin': 'zero', 'width': w, 'height': h,
             'scale_source': 'synthetic', 'px_nm': .5}))
    return out


def _normalize(frame, base, params, frozen=None):
    if frame.empty:
        return frame.copy(), {}
    pooled = cdqc.cohort_stats(base, params=params) if not base.empty else None
    used, parts = {}, []
    for cat, group in frame.groupby('category', sort=False, dropna=False):
        st = frozen.get(str(cat)) if frozen is not None else None
        if frozen is None:
            b = base[base['category'] == cat]
            if len(b):
                original_stats = cdqc.cohort_stats(b, params=params)
                st = cdqc.floor_stats(original_stats, pooled, params)
                st['raw_features'] = original_stats['features']
                st['valid_counts'] = {f.name: int(np.isfinite(pd.to_numeric(b[f.name], errors='coerce')).sum())
                                      for f in cdqc.REGISTRY if f.name in b}
        if st is None:
            z = group.copy()
            for feature in cdqc.REGISTRY:
                if feature.name in z and feature.kind in ('z', 'bool', 'match'):
                    z['z_' + feature.name] = np.nan
            z['baseline_available'] = False
        else:
            used[str(cat)] = st
            z = cdqc.apply_z(group, st, params)
            for col in z:
                if col.startswith(('z_', 'zs_')):
                    z[col] = z[col].clip(-params.z_cap, params.z_cap)
            z['baseline_available'] = True
        if any('z_' + f.name in z for f in cdqc.REGISTRY):
            z = z.join(cdqc.aggregate_z(z, params=params)).join(cdqc.top_feature(z, params=params))
        else:
            for column in cdqc.agg_columns() + ['top_z']:
                z[column] = np.nan
            z['n_z_valid'], z['top_feature'], z['reason_code'] = 0, '', ''
        parts.append(z)
    return pd.concat(parts).sort_index(), used


def analyze(datasets, params=None, baseline_ids=None, frozen_stats=None, progress=None) -> Analysis:
    datasets = list(datasets)
    params = validate_params(params)
    ids = [d.image_id for d in datasets]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('데이터셋이 필요하며 image_id는 중복되면 안 됩니다.')
    if frozen_stats is not None:
        if frozen_stats.get('schema_version') != 1 or frozen_stats.get('cdqc_version') != cdqc.__version__:
            raise ValueError('기준 통계의 스키마 또는 cdqc 버전이 다릅니다.')
        if _json_safe(frozen_stats.get('params')) != _json_safe(asdict(params)):
            raise ValueError('기준 통계와 현재 계산 파라미터가 다릅니다. 기준과 같은 설정을 사용하세요.')
        baseline = list(frozen_stats.get('baseline_ids', []))
    else:
        baseline = ids if baseline_ids is None else list(baseline_ids)
        if not baseline or not set(baseline) <= set(ids):
            raise ValueError('정상 기준 이미지를 한 장 이상 선택해야 합니다. 알 수 없는 ID는 사용할 수 없습니다.')
    issues = [dict(issue) for d in datasets for issue in d.issues]
    if frozen_stats is None and baseline_ids is None:
        issues.append(_issue('EXPLORATORY_BASELINE', '전체 입력으로 기준을 계산하는 탐색 모드입니다. 불량 혼입 시 기준이 변합니다.'))
    all_l3, all_l2, all_l1 = [], [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for dataset_index, dataset in enumerate(datasets):
            if progress:
                progress(f'{dataset.image_id}: 피쳐 계산', dataset_index, len(datasets))
            maps = cdqc.mask_maps(dataset.labelmap, dataset.img) if dataset.labelmap is not None else None
            for cat, records in dataset.records.groupby('category', sort=False, dropna=False):
                valid = records[records['analysis_valid']].copy()
                shell = records[['source_row', 'cd_index', 'category', 'analysis_valid', 'sx', 'sy', 'ex', 'ey', 'value_nm']].copy()
                shell['image_id'] = dataset.image_id
                for target, source in (('s_x', 'sx'), ('s_y', 'sy'), ('e_x', 'ex'), ('e_y', 'ey')):
                    shell[target] = shell[source]
                shell['mid_x'] = (shell.sx + shell.ex) / 2
                shell['mid_y'] = (shell.sy + shell.ey) / 2
                shell['px_nm'] = dataset.px_nm
                if len(valid):
                    s = valid[['sx', 'sy']].to_numpy(float)
                    e = valid[['ex', 'ey']].to_numpy(float)
                    f = cdqc.extract_l3(dataset.img, s, e, dataset.px_nm,
                        value_nm=valid['value_nm'].to_numpy(float), params=params)
                    if maps is not None:
                        f = f.join(cdqc.boundary_features(maps, s, e, dataset.px_nm, params))
                    f.index = valid.index
                    for column in f:
                        if column in shell:
                            shell.loc[valid.index, column] = f[column]
                        else:
                            shell[column] = f[column]
                    l2 = cdqc.extract_l2(f, params=params)
                    l2['category'], l2['image_id'] = cat, dataset.image_id
                    l2['input_count'], l2['excluded_count'] = len(records), len(records)-len(valid)
                    all_l2.append(l2)
                    if len(valid) < len(records):
                        issues.append(_issue('SEQUENCE_GAPS', '제외된 행 사이의 유효 CD를 이어서 시퀀스 피쳐를 계산했습니다. 원본 행과 순서를 확인하세요.', dataset.image_id, category=cat))
                all_l3.append(shell)
            l1 = cdqc.extract_l1(dataset.img, params)
            l1.pop('hist', None)
            all_l1.append(dict(l1, image_id=dataset.image_id, category='__all_images__'))
        for warning in caught:
            issues.append(_issue('COMPUTATION_WARNING', str(warning.message), severity='warning'))
    frames = [pd.concat(all_l3, ignore_index=True),
              pd.concat(all_l2, ignore_index=True) if all_l2 else pd.DataFrame(),
              pd.DataFrame(all_l1)]
    stats = {'schema_version': 1, 'cdqc_version': cdqc.__version__, 'params': asdict(params),
             'baseline_ids': baseline, 'created_at': datetime.now(timezone.utc).isoformat()}
    result = []
    for level, frame in zip(('l3', 'l2', 'l1'), frames):
        valid = frame['analysis_valid'].fillna(False).astype(bool) if 'analysis_valid' in frame else pd.Series(True, index=frame.index)
        base = frame[valid & frame['image_id'].isin(baseline)] if not frame.empty else frame
        norm, level_stats = _normalize(frame, base, params,
                                      frozen_stats.get(level, {}) if frozen_stats is not None else None)
        if not norm.empty:
            if 'analysis_valid' in norm:
                excluded = ~norm['analysis_valid'].fillna(False).astype(bool)
                for col in norm:
                    if col.startswith(('z_', 'zs_')) or col in ('top_z',):
                        norm.loc[excluded, col] = np.nan
                norm.loc[excluded, 'n_z_valid'] = 0
                norm.loc[excluded, 'top_feature'] = ''
                norm.loc[excluded, 'reason_code'] = ''
            for cat in norm.loc[~norm['baseline_available'], 'category'].unique():
                issues.append(_issue('MISSING_BASELINE', f'{level}: {cat} 기준 없음. z 계산 안 함.', category=str(cat)))
            for cat, st in level_stats.items():
                if st['n'] < params.min_cohort_n:
                    issues.append(_issue('SMALL_BASELINE', f'{level}: {cat} 기준 n={st["n"]} < {params.min_cohort_n}. 연속 피쳐 z가 비어 있을 수 있습니다.', category=str(cat)))
        result.append(norm)
        stats[level] = frozen_stats.get(level, {}) if frozen_stats is not None else level_stats
    l3, l2, l1 = result
    summary = []
    for cat, group in l3.groupby('category', sort=False):
        summary.append(dict(category=cat, n_images=group['image_id'].nunique(), n_cd=len(group),
            n_valid=int(group['analysis_valid'].sum()), n_excluded=int((~group['analysis_valid']).sum()),
            n_baseline=int((group['image_id'].isin(baseline) & group['analysis_valid']).sum()),
            z_max=group['z_max'].max() if 'z_max' in group else np.nan,
            cd_median_nm=group['cd_nm'].median() if 'cd_nm' in group else np.nan))
    if progress:
        progress('분석 완료', len(datasets), len(datasets))
    return Analysis(datasets, l3, l2, l1, pd.DataFrame(summary), stats, params, issues,
                    baseline, uuid.uuid4().hex[:12])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_stats(analysis: Analysis) -> str:
    return json.dumps(_json_safe(analysis.stats), ensure_ascii=False, indent=2, allow_nan=False)


def import_stats(text) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('지원하지 않는 기준 통계 파일입니다.')
    for level in ('l3', 'l2', 'l1'):
        if not isinstance(value.get(level), dict):
            raise ValueError(f'기준 통계에 {level} 정보가 없습니다.')
        for stats in value[level].values():
            for container in ('features', 'raw_features'):
                for feature in stats.get(container, {}).values():
                    for key in ('median', 'mad'):
                        feature[key] = np.nan if feature.get(key) is None else float(feature[key])
    return value


def registry_table() -> pd.DataFrame:
    return pd.DataFrame([asdict(feature) for feature in cdqc.REGISTRY])


def quality_table(analysis: Analysis) -> pd.DataFrame:
    return pd.DataFrame(analysis.issues, columns=list(dict.fromkeys(
        ['severity', 'code', 'image_id', 'message'] + [k for issue in analysis.issues for k in issue])))


def feature_diagnostics(analysis: Analysis, row_index, level='l3') -> pd.DataFrame:
    frame = getattr(analysis, level)
    row = frame.loc[row_index]
    st = analysis.stats.get(level, {}).get(str(row['category']), {})
    items = []
    for feature in cdqc.REGISTRY:
        if feature.name not in row:
            continue
        name = feature.name
        ref = st.get('features', {}).get(name, {})
        original_ref = st.get('raw_features', {}).get(name, {})
        mad = ref.get('mad', np.nan)
        scale = max(1.4826*mad, analysis.params.mad_floor(name)) if np.isfinite(mad) else np.nan
        items.append(dict(feature=name, raw=row[name], z=row.get('z_' + name, np.nan),
            signed_z=row.get('zs_' + name, np.nan), reference_median=ref.get('median', np.nan),
            reference_mad=mad, reference_mode=st.get('modes', {}).get(name, np.nan),
            raw_reference_mad=original_ref.get('mad', np.nan),
            scale=scale, reference_n=st.get('n', 0), reference_valid_n=st.get('valid_counts', {}).get(name, 0),
            floor_dominates=bool(np.isfinite(scale) and np.isfinite(original_ref.get('mad', np.nan))
                                 and scale > 1.4826*original_ref['mad']),
            space='log(x+0.001)' if name in analysis.params.log_features else 'raw',
            direction=analysis.params.direction_overrides.get(name, feature.worse_when),
            enabled=feature.enabled_default, kind=feature.kind, reason=feature.reason,
            status=('input_excluded' if not row.get('analysis_valid', True) else
                    'raw_only' if feature.kind not in ('z', 'bool', 'match') else
                    'missing_value' if pd.isna(row[name]) else
                    'missing_baseline' if not row.get('baseline_available', False) else
                    'insufficient_reference' if feature.kind == 'z' and not np.isfinite(ref.get('median', np.nan)) else
                    'available'),
            description=feature.desc))
    return pd.DataFrame(items)


def profile_table(dataset: Dataset, category, cd_index, params=None) -> pd.DataFrame:
    """Use the exact full valid sequence/tangents and sampler used by extract_l3."""
    params = params or cdqc.Params()
    seq = dataset.records[(dataset.records['category'] == category) & dataset.records['analysis_valid']]
    where = np.flatnonzero(seq['cd_index'].to_numpy() == cd_index)
    if not len(where):
        return pd.DataFrame(columns=['s_px', 's_nm', 'intensity', 'gradient'])
    s, e = seq[['sx', 'sy']].to_numpy(float), seq[['ex', 'ey']].to_numpy(float)
    tangents = mean_edge_tangent(unit_tangents(s), unit_tangents(e))
    profile = sample_ribbon_profiles(dataset.img, s, e, tangents, params.view()['sampling'])[int(where[0])]
    smooth = gaussian_filter1d(profile.p, max(params.grad_sigma_px / DS, 0.01))
    gradient = gradient_of(profile, params.grad_sigma_px)
    frame = pd.DataFrame(dict(s_px=profile.s, s_nm=profile.s*dataset.px_nm,
                              intensity=profile.p, smoothed=smooth, gradient=gradient))
    frame.attrs.update(length_px=profile.L, length_nm=profile.L*dataset.px_nm,
        noise_sigma=profile.sigma, tangent=profile.tangent.tolist(), s_endpoint_px=0.,
        e_endpoint_px=profile.L, sampling_step_px=DS, grad_sigma_px=params.grad_sigma_px,
        window_half_px=float(np.clip(params.win_frac*profile.L, params.win_min_px, params.win_max_px))/2)
    offsets = np.arange(-params.ribbon_half_w, params.ribbon_half_w + 1)
    points = (s[int(where[0])][None, None, :] + profile.s[None, :, None]*profile.u
              + offsets[:, None, None]*profile.tangent)
    h, w = dataset.img.shape
    outside = (points[..., 0] < 0) | (points[..., 0] > w-1) | (points[..., 1] < 0) | (points[..., 1] > h-1)
    frame.attrs['sampling_outside_fraction'] = float(outside.mean())
    frame.attrs['outside_sampling_mode'] = 'nearest (core sampler)'
    half = frame.attrs['window_half_px']
    frame.attrs['window_s'] = [-half, half]
    frame.attrs['window_e'] = [profile.L-half, profile.L+half]
    return frame
