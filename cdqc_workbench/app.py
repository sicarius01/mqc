"""Local diagnostic workbench. Display controls never run cohort analysis."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cdqc.params import Params
from cdqc_workbench import backend as be
from cdqc_workbench import discovery as ds
from cdqc_workbench.plots import image_overlay, profile_plot


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def validated_import(value):
    config = ds.validate_config(value)
    config["params"] = asdict(be.validate_params(config.get("params")))
    return config


def role_config_from_frame(data):
    output = {}
    for record in data.to_dict("records"):
        values = {}
        for name in ("directory", "extension", "suffix"):
            cell = record.get(name)
            value = "" if cell is None or pd.isna(cell) else str(cell).strip()
            values[name] = None if name == "directory" and value.lower() in ("", "none") else value
        output[record["role"]] = values
    return output


def frame(value):
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame()


def table(value, *, key=None, height=350):
    data = frame(value)
    if data.empty:
        st.info("표시할 항목이 없습니다.")
        return
    # Arrow cannot infer mixed object cells such as boolean/float diagnostic values.
    show = data.copy()
    for col in [c for c in show if pd.api.types.is_object_dtype(show[c].dtype)]:
        show[col] = show[col].map(lambda x: "NaN · 미산출" if x is None or (np.isscalar(x) and pd.isna(x))
                                  else json_text(x) if isinstance(x, (dict, list, tuple)) else str(x))
    st.dataframe(show, use_container_width=True, height=height, hide_index=True, key=key)


def issue_error(exc, context):
    st.error(f"{context}: {type(exc).__name__}: {exc}")
    st.session_state["last_error"] = f"{context}\n{traceback.format_exc()}"
    # A container also works when this error is reported inside an expander.
    with st.container(border=True):
        st.caption("오류 상세 · 로컬 진단용")
        st.code(st.session_state.last_error)


def init_state():
    if "config" not in st.session_state:
        try:
            st.session_state.config = ds.load_config()
        except (ValueError, OSError) as exc:
            st.session_state.config = ds.default_config()
            st.session_state.startup_error = str(exc)
    st.session_state.setdefault("datasets", [])
    st.session_state.setdefault("matches", pd.DataFrame())
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("params_json", json_text(st.session_state.config.get("params", asdict(Params()))))
    st.session_state.setdefault("frozen_stats", None)
    st.session_state.setdefault("run_count", 0)


def configure():
    st.subheader("D0 · 폴더를 지정하고 파일 연결 규칙을 확인하세요")
    st.caption("하위 폴더까지 찾은 뒤 같은 측정 묶음의 TIF · 측정표 · 세그멘테이션 PNG · DM3를 연결합니다. 규칙은 이 화면에서 바꾸고 저장할 수 있습니다.")
    st.caption("XLSX는 이 PC의 Microsoft Excel을 창 없이 실행해 읽습니다 (read_nasca_csv). Excel 설치가 필요합니다.")
    if st.session_state.get("startup_error"):
        st.warning("저장 설정을 읽지 못해 기본값을 사용합니다: " + st.session_state.startup_error)
    cfg = dict(st.session_state.config)
    root = st.text_input("데이터 루트 폴더", cfg.get("root", ""), placeholder=r"D:\TEM_data", key="root_path")
    roles = cfg.get("roles", {"dm3": {"directory": None, "extension": ".dm3", "suffix": ""},
                             "image": {"directory": None, "extension": ".tif", "suffix": ""},
                             "seg": {"directory": "result", "extension": ".png", "suffix": ""},
                             "table": {"directory": "result", "extension": ".xlsx", "suffix": "_Result"}})
    role_rows = pd.DataFrame([{"role": role, "directory": settings.get("directory") or "",
                               "extension": settings.get("extension", ""), "suffix": settings.get("suffix", "")}
                              for role, settings in roles.items()])
    st.markdown("**파일 종류별 규칙** · 각 측정 폴더에서 아래 규칙으로 네 파일을 연결합니다.")
    edited_roles = st.data_editor(role_rows, hide_index=True, use_container_width=True, disabled=["role"],
        column_config={"role": st.column_config.TextColumn("종류 (dm3 / image / seg / table)"),
                       "directory": st.column_config.TextColumn("하위 디렉토리 · 비우면 측정 폴더"),
                       "extension": st.column_config.TextColumn("확장자"),
                       "suffix": st.column_config.TextColumn("파일명 접미어 · 비워도 됨")}, key="role_rules")
    st.caption(r"예: abc.dm3 + abc.tif + result\abc.png + result\abc_Result.xlsx → abc 한 묶음. None 또는 빈 디렉토리는 측정 폴더 자체입니다.")
    with st.expander("탐색 범위 · 고급 연결 방식 · 좌표 옵션", expanded=False):
        col1, col2 = st.columns([3, 2])
        with col1:
            patterns = {}
            for role, label in [("image", "TIF 이미지"), ("table", "측정표 XLSX / CSV"),
                                ("seg", "세그멘테이션 PNG"), ("dm3", "픽셀 크기 DM3")]:
                patterns[role] = st.text_input(f"{label} 파일 패턴", cfg["patterns"].get(role, ""),
                                                key="pattern_" + role,
                                                help="여러 패턴은 세미콜론으로 구분합니다. 예: *.tif;*.tiff 또는 **/images/*.tif")
        with col2:
            recursive = st.checkbox("모든 하위 폴더 탐색", cfg.get("recursive", True))
            modes = ["layout", "stem", "folder", "regex"]
            match_mode = st.selectbox("파일 묶음 연결 방식", modes, index=modes.index(cfg.get("match_mode", "stem")),
                                      format_func=lambda v: {"layout": "위 표의 디렉토리 / 확장자 / 접미어 규칙", "stem": "확장자를 뺀 파일명 일치", "folder": "같은 폴더 안의 파일", "regex": "이름에서 공통 키 추출 (정규식)"}[v])
            key_source = st.selectbox("연결 키를 읽을 위치", ["stem", "relative_path"],
                                      index=["stem", "relative_path"].index(cfg.get("key_source", "stem")),
                                      format_func=lambda v: "확장자를 뺀 파일명" if v == "stem" else "루트 기준 상대 경로")
            require_seg = st.checkbox("세그멘테이션 필수", cfg.get("require_seg", True))
            require_dm3 = st.checkbox("DM3 필수", cfg.get("require_dm3", True))
            px_override = st.number_input("공통 픽셀 크기 (nm/px) · 0이면 DM3 사용", min_value=0.0,
                                          value=float(cfg.get("px_nm") or 0), format="%.6f")
        regex = st.text_input("공통 키 정규식 · regex 방식에서 사용", cfg.get("key_regex", ""),
                              placeholder=r"(?P<key>.+?)(?:_image|_seg|_measure)?$",
                              help="(?P<key>...)에 들어온 문자가 같은 파일들을 묶습니다. 아래 역할별 정규식으로 서로 다른 접미사를 제거할 수 있습니다.")
        with st.container(border=True):
            st.markdown("**고급 규칙 · 좌표 변환 · 열 이름 매핑**")
            st.markdown("파일명 규칙을 설명받은 뒤 **역할별 정규식**을 여기에 붙여 넣을 수 있습니다. 좌표 변경은 다음 분석에 적용됩니다.")
            role_regex_json = st.text_area("역할별 정규식 JSON (image / table / seg / dm3)",
                                           json_text(cfg.get("role_key_regex", {})), height=100)
            ignored = st.text_input("무시할 파일 패턴 · 세미콜론 구분", ";".join(cfg.get("ignored_patterns", [])))
            options_json = st.text_area("로딩·좌표 옵션 JSON", json_text(cfg.get("options", {})), height=190,
                                        help='예: {"coordinate_mode":"pixel", "origin":"zero", "y_flip":false, "offset_x":0, "offset_y":0, "mapping":{"sx":"Start X", "sy":"Start Y", "ex":"End X", "ey":"End Y"}}')
            st.caption("coordinate_mode: auto / center_m / pixel · origin: zero / one / center_half / center_pixel · y_flip · swap_xy · offset_x/y(px) · order_col · value_unit · seg_restore · seg_resize. mapping 키: sx, sy, ex, ey, category, value_nm, unit. 탐색 후 열 미리보기에서 실제 이름을 확인하세요.")
        offsets = st.columns(2)
        offset_x = offsets[0].number_input("좌표 X 보정 (px) · 양수는 오른쪽", value=float(cfg.get("options", {}).get("offset_x", 0)), step=.5)
        offset_y = offsets[1].number_input("좌표 Y 보정 (px) · 양수는 아래쪽", value=float(cfg.get("options", {}).get("offset_y", 0)), step=.5)
    try:
        role_config = role_config_from_frame(edited_roles)
        cfg.update(root=root, recursive=recursive, patterns=patterns, match_mode=match_mode, roles=role_config,
                   key_source=key_source, key_regex=regex, require_seg=require_seg,
                   require_dm3=require_dm3, px_nm=px_override or None,
                   role_key_regex=json.loads(role_regex_json), ignored_patterns=[x.strip() for x in ignored.split(";") if x.strip()],
                   options=json.loads(options_json))
        if not isinstance(cfg["options"], dict):
            raise ValueError("로딩·좌표 옵션은 JSON 객체여야 합니다.")
        cfg["options"].update(offset_x=offset_x, offset_y=offset_y)
        cfg = ds.validate_config(cfg)
    except (ValueError, TypeError) as exc:
        st.error(f"설정 형식 확인: {exc}")
        return None, False
    a, b, c = st.columns(3)
    with a:
        if st.button("폴더 규칙 저장", use_container_width=True):
            try:
                cfg["params"] = asdict(be.validate_params(json.loads(st.session_state.params_json)))
                ds.save_config(cfg)
                st.session_state.config = cfg
                st.success("다음 실행에서도 이 규칙을 사용합니다.")
            except (ValueError, OSError) as exc:
                issue_error(exc, "설정 저장 실패")
    with b:
        scan = st.button("탐색 · 연결 미리보기", use_container_width=True)
    with c:
        auto = st.button("탐색 후 전체 분석", type="primary", use_container_width=True)
    with st.expander("설정 파일 가져오기 / 내보내기"):
        st.caption(f"기본 저장 위치: {ds.DEFAULT_CONFIG_PATH} · 설정에는 입력한 로컬 경로가 포함됩니다.")
        try:
            download_cfg = dict(cfg, params=asdict(be.validate_params(json.loads(st.session_state.params_json))))
            st.download_button("현재 폴더 규칙 JSON 저장", json_text(download_cfg), "workbench-config.json", "application/json")
        except (ValueError, TypeError) as exc:
            st.warning(f"설정 내보내기 전에 파라미터를 수정하세요: {exc}")
        upload = st.file_uploader("다른 설정 JSON 불러오기", type="json", key="config_upload")
        if st.button("불러온 설정 적용", disabled=upload is None):
            try:
                st.session_state.pending_import_config = validated_import(json.loads(upload.getvalue()))
                st.rerun()
            except (ValueError, TypeError) as exc:
                issue_error(exc, "설정 가져오기 실패")
    if scan or auto:
        st.session_state.scan_attempted = True
        try:
            with st.spinner("폴더 탐색 및 연결 키 확인 중…"):
                st.session_state.matches = ds.scan_directory(cfg)
                st.session_state.match_revision = st.session_state.get("match_revision", 0) + 1
                st.session_state.scanned_config = cfg
                st.session_state.config = cfg
            if st.session_state.matches.empty:
                st.warning("일치하는 파일이 없습니다. 폴더와 파일 패턴을 확인하세요.")
                auto = False
        except Exception as exc:
            issue_error(exc, "탐색 실패")
            auto = False
    return cfg, auto


def matches_preview():
    matches = st.session_state.matches
    if st.session_state.get("scan_attempted"):
        st.caption(f"탐색 파일 {matches.attrs.get('files_scanned', 0)} · 연결 묶음 {len(matches)} · 준비 완료 {int((matches.get('status', pd.Series(dtype=str)) == 'ready').sum())}")
        with st.expander("탐색 진단 · 패턴 불일치 / 접미어 불일치 / 읽기 실패", expanded=matches.empty):
            st.json(matches.attrs, expanded=False)
            table(matches.attrs.get("unmatched_details", []), height=200)
    if matches.empty:
        return matches
    st.markdown("**연결 결과 · 후보가 여러 개이거나 필수 파일이 빠진 묶음은 자동 선택하지 않습니다.**")
    st.caption("경로와 픽셀 크기는 셀을 더블클릭해 수정하고 enabled를 켤 수 있습니다. 수정한 경로는 실제 로딩 때 다시 검사합니다. 연결 방식이나 규칙을 바꾸면 다시 탐색하세요.")
    cols = [c for c in ["enabled", "image_id", "image_path", "table_path", "seg_path", "dm3_path", "px_nm", "status", "issues"] if c in matches]
    edited = st.data_editor(matches[cols], use_container_width=True, hide_index=True,
                            disabled=[c for c in ["image_id", "status", "issues"] if c in cols],
                            column_config={"enabled": st.column_config.CheckboxColumn("분석 포함")},
                            key=f"matches_editor_{st.session_state.get('match_revision', 0)}")
    with st.expander("왜 이 파일들이 연결되었나요? · 모든 후보와 추출 키"):
        table(matches, height=300)
        if matches.attrs:
            st.json(matches.attrs, expanded=False)
    enabled = edited[edited.enabled.fillna(False).astype(bool)]
    st.caption(f"전체 묶음 {len(edited)} · 분석 포함 {len(enabled)} · 제외 {len(edited) - len(enabled)}")
    paths = edited.table_path.dropna().astype(str)
    paths = paths[paths.str.len() > 0].unique().tolist()
    with st.expander("측정표 열 미리보기 · 좌표 매핑 확인"):
        if paths:
            path = st.selectbox("미리 볼 측정표", paths)
            if st.button("열 이름과 첫 20행 읽기"):
                try:
                    raw = be.read_table(path)
                    st.session_state.table_preview = (path, raw.head(20), list(raw.columns))
                except Exception as exc:
                    issue_error(exc, "측정표 미리보기 실패")
            preview = st.session_state.get("table_preview")
            if preview and preview[0] == path:
                st.code(json_text(preview[2]))
                table(preview[1])
    return edited


def analysis_settings(ids):
    with st.expander("분석 기준 · 정상 기준 선택 · 계산 파라미터", expanded=False):
        mode = st.radio("비교 기준", ["탐색: 입력 전체", "선택한 정상 이미지", "저장한 기준 JSON"], horizontal=True)
        chosen = st.multiselect("정상 기준 이미지 ID", ids, default=[], disabled=mode != "선택한 정상 이미지")
        st.caption("입력 전체 기준은 배치 구성이 바뀌면 점수도 달라집니다. 고정 정상 기준으로 비교하려면 정상 이미지를 선택해 기준 JSON을 저장하세요.")
        upload = st.file_uploader("기준 통계 JSON", type="json", key="stats_upload", disabled=mode != "저장한 기준 JSON")
        # The active upload owns the reference; never reuse a removed file's cache.
        st.session_state.frozen_stats = None
        if upload is not None and mode == "저장한 기준 JSON":
            try:
                st.session_state.frozen_stats = be.import_stats(upload.getvalue().decode("utf-8-sig"))
                st.success("기준 통계 형식 확인 완료")
            except Exception as exc:
                st.session_state.frozen_stats = None
                issue_error(exc, "기준 통계 읽기 실패")
        st.text_area("전체 계산 파라미터 JSON · 다음 분석에 적용", key="params_json", height=290)
        st.caption("탐색 창, 평활화, MAD 하한, 최소 기준 표본 수, 방향, z 상한까지 모두 조정할 수 있습니다. 기본값을 기준으로 한 번에 한 항목씩 변경하세요.")
        if st.button("파라미터 기본값 복원"):
            st.session_state.params_reset = True
            st.rerun()
    return mode, chosen


def run_analysis(cfg, matches, mode, chosen, *, demo=False):
    load_issues = []
    manifest = ([] if demo or matches.empty else
                matches[matches.enabled.fillna(False).astype(bool)].to_dict("records"))
    attempt = {"config": deepcopy(cfg), "manifest": deepcopy(manifest), "mode": mode,
               "params_json": st.session_state.params_json, "load_issues": load_issues}
    try:
        params = be.validate_params(json.loads(st.session_state.params_json))
        frozen = st.session_state.frozen_stats if mode == "저장한 기준 JSON" else None
        if mode == "저장한 기준 JSON" and frozen is None:
            raise ValueError("저장한 기준 JSON을 먼저 불러오세요.")
        if mode == "선택한 정상 이미지" and not chosen:
            raise ValueError("정상 기준 이미지를 하나 이상 선택하세요.")
        datasets = []
        with st.status("분석 실행 중", expanded=True) as status:
            if demo:
                datasets = be.demo_datasets()
                st.write(f"합성 데모 {len(datasets)}개 이미지 준비")
            elif not matches.empty:
                rows = matches[matches.enabled.fillna(False).astype(bool)]
                if rows.empty:
                    raise ValueError("연결 미리보기에서 분석 포함을 선택한 묶음이 없습니다.")
                progress = st.progress(0.0)
                for i, row in enumerate(rows.to_dict("records")):
                    st.write(f"[{i + 1}/{len(rows)}] {row['image_id']} 로딩")
                    clean = lambda value: None if value is None or pd.isna(value) or str(value).strip() == "" else value
                    try:
                        datasets.append(be.load_dataset(image_path=clean(row.get("image_path")),
                            table_path=clean(row.get("table_path")), seg_path=clean(row.get("seg_path")),
                            dm3_path=clean(row.get("dm3_path")), px_nm=clean(row.get("px_nm")),
                            options=cfg.get("options", {}), image_id=str(row["image_id"])))
                    except Exception as exc:
                        load_issues.append({"image_id": row["image_id"], "stage": "load", "message": f"{type(exc).__name__}: {exc}"})
                        st.warning(f"{row['image_id']} 로딩 실패: {exc}")
                    progress.progress((i + 1) / len(rows))
            else:
                datasets = st.session_state.datasets
                if not datasets or not all(d.metadata.get("demo") for d in datasets):
                    raise ValueError("실제 데이터는 새 연결 결과에서 분석 포함을 선택하세요. 이전 데이터로 자동 대체하지 않습니다.")
            if not datasets:
                raise ValueError("분석 가능한 데이터가 없습니다. 연결 결과와 로딩 오류를 확인하세요.")
            st.write("L3 개별 CD → L2 시퀀스 → L1 이미지 특징 및 기준 대비 점수 계산")
            analysis_progress = st.progress(0.0)
            analysis_message = st.empty()
            def report(message, current, total):
                analysis_message.write(message)
                analysis_progress.progress(min(1.0, current / max(total, 1)))
            result = be.analyze(datasets, params=params, baseline_ids=chosen if mode == "선택한 정상 이미지" else None,
                                frozen_stats=frozen, progress=report)
            synthetic = all(d.metadata.get("demo") for d in datasets)
            # Commit the result and its provenance together, only after success.
            st.session_state.update(datasets=datasets, analysis=result,
                load_issues=load_issues, run_count=st.session_state.run_count + 1,
                run_config={"source": "synthetic_demo"} if synthetic else deepcopy(cfg),
                run_manifest=deepcopy(manifest),
                run_scan_config={} if synthetic else deepcopy(st.session_state.get("scanned_config", {})),
                run_mode=mode, last_error="", failed_attempt=None)
            if demo:
                st.session_state.scan_attempted = False
            for key in ("selected_image", "selected_category", "selected_cd"):
                st.session_state.pop(key, None)
            status.update(label=f"분석 완료 · 이미지 {len(datasets)}개 · 로딩 실패 {len(load_issues)}개", state="complete", expanded=False)
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        st.session_state.failed_attempt = attempt
        issue_error(exc, "분석 실패 · 이전 성공 결과는 유지됩니다")


def overview(result):
    st.subheader("D1 · 어디부터 볼지 정하기")
    l3 = result.l3
    score = pd.to_numeric(l3.get("z_max", pd.Series(dtype=float)), errors="coerce")
    threshold = st.number_input("관찰용 z 기준 · 판정 정책이 아닌 표시 필터", min_value=0.0, value=3.0, step=.5)
    cells = st.columns(4)
    cells[0].metric("분석 이미지", len(result.datasets))
    cells[1].metric("CD 행", len(l3))
    cells[2].metric(f"z_max ≥ {threshold:g}", int((score >= threshold).sum()))
    cells[3].metric("z_max 미산출", int(score.isna().sum()))
    st.info("점수는 정상 기준에서 벗어난 정도를 관찰하는 값입니다. 30 같은 상한값은 포화일 수 있으며 심각도 순위로 해석하지 않습니다. NaN은 미산출이며 정상 0점과 다릅니다.")
    quality = be.quality_table(result)
    table(quality, height=260)
    st.markdown("**전체 CD 결과 · 정렬 / 검색 후 D2에서 같은 원본 행을 선택하세요**")
    filter_images = st.multiselect("요약 표에 표시할 이미지 · 비우면 전체", l3.image_id.astype(str).unique().tolist(), key="overview_images")
    query = st.text_input("결과 검색 · 이미지 / 카테고리 / 원본 행", key="overview_query")
    high = st.checkbox("관찰 기준 이상인 행만 보기")
    shown = l3.copy()
    if filter_images:
        shown = shown[shown.image_id.astype(str).isin(filter_images)]
    if high and "z_max" in shown:
        shown = shown[pd.to_numeric(shown.z_max, errors="coerce") >= threshold]
    if query:
        cols = [x for x in ["image_id", "category", "source_row", "cd_index"] if x in shown]
        mask = shown[cols].astype(str).apply(lambda x: x.str.contains(query, case=False, regex=False)).any(axis=1)
        shown = shown[mask]
    front = [c for c in ["image_id", "category", "source_row", "cd_index", "analysis_valid", "z_max", "z_top2_kth", "n_z_valid", "cd_nm"] if c in shown]
    event = st.dataframe(shown[front + [c for c in shown if c not in front]], height=430, use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row", key=f"overview_rows_{hash(tuple(shown.index))}")
    selected_rows = event["selection"]["rows"]
    if selected_rows:
        linked_preview(result, shown.index[selected_rows[0]], "overview")


def linked_preview(result, index, prefix):
    """Show a clicked row by its original analysis index, never display position."""
    row = result.l3.loc[index]
    dataset = next(d for d in result.datasets if str(d.image_id) == str(row.image_id))
    peers = result.l3[(result.l3.image_id == row.image_id) & (result.l3.category == row.category)]
    st.markdown(f"**선택 값의 실제 위치** · 이미지 `{row.image_id}` · 카테고리 `{row.category}` · 원본 행 `{row.source_row}`")
    st.plotly_chart(image_overlay(dataset, peers, row, focus=True), use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False}, key=prefix + "_value_overlay")
    st.button("이 값을 D2 / D3 상세에 적용", key=prefix + "_apply",
              on_click=lambda: st.session_state.update(pending_focus=(str(row.image_id), str(row.category), index)))
    with st.expander("선택 값의 특징별 근거"):
        table(be.feature_diagnostics(result, index), height=300)


def selected_source_index(candidates, selection_rows):
    """Map UI positions back to stable source indices; reject stale selections."""
    if not selection_rows:
        return None
    position = selection_rows[0]
    return candidates.index[position] if isinstance(position, int) and 0 <= position < len(candidates) else None


def select_cd(result):
    st.subheader("D2 · 이미지에서 문제 CD를 확인하기")
    a, b, c = st.columns([2, 2, 3])
    images = list(dict.fromkeys(result.l3.image_id.astype(str)))
    with a:
        image_id = st.selectbox("이미지", images, key="selected_image")
    rows = result.l3[result.l3.image_id.astype(str) == image_id]
    with b:
        category = st.selectbox("카테고리", rows.category.astype(str).unique(), key="selected_category")
    rows = rows[rows.category.astype(str) == category]
    with c:
        selected_index = st.selectbox("CD · 원본 행 번호는 필터와 정렬 후에도 유지", rows.index.tolist(), key="selected_cd",
                                     format_func=lambda i, data=result.l3: f"원본 행 {data.loc[i, 'source_row']} · 순서 {data.loc[i, 'cd_index']} · z={data.loc[i].get('z_max', np.nan):.3g}")
    row = result.l3.loc[selected_index]
    dataset = next(d for d in result.datasets if str(d.image_id) == image_id)
    controls = st.columns([1, 1, 2])
    show_mask = controls[0].checkbox("세그멘테이션 겹치기", disabled=dataset.labelmap is None)
    focus = controls[1].checkbox("선택 CD 확대", value=True)
    contrast = controls[2].slider("표시 명암 범위 · 계산값에는 영향 없음", 0, 255, (0, 255))
    if dataset.img is None:
        st.info("이미지 미제공: 좌표만 표시합니다. 이미지 증거는 미산출이며 실패 점수가 아닙니다.")
    st.plotly_chart(image_overlay(dataset, rows, row, show_mask, focus, contrast), use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False}, key="overlay")
    st.caption("청록색: 같은 카테고리 CD · 노란색: 선택 CD와 보고 S/E. 선택 CD 확대는 원본 픽셀 해상도, 전체 보기는 표시용 축소입니다. 드래그 이동, 휠 확대, 더블클릭 복원. 좌표 원점은 좌상단, x=열, y=행입니다.")
    summary = {k: row.get(k) for k in ["source_row", "cd_index", "cd_nm", "s_x", "s_y", "e_x", "e_y", "px_nm", "z_max", "z_top2_kth", "n_z_valid"]}
    table(summary, height=115)
    with st.expander("입력 파일 · 검출 열 · 좌표 변환 메타데이터"):
        st.json(dataset.metadata, expanded=2)
        table(dataset.records, height=230)
    return dataset, row, selected_index, rows


def diagnostics(result, dataset, row, index, rows):
    st.subheader("D3 · 이 CD의 점수가 나온 이유")
    diag = be.feature_diagnostics(result, index)
    text = st.text_input("특징 이름 / 설명 검색", placeholder="delta, cnr, angle, 경계…")
    if text and not diag.empty:
        mask = diag.astype(str).apply(lambda x: x.str.contains(text, case=False, regex=False)).any(axis=1)
        diag = diag[mask]
    st.caption("raw: 원값 · signed z: 기준보다 높고 낮은 방향 · directed z: 특징의 이상 방향 적용 · 기준 통계: 정규화에 실제 사용된 값. 표 오른쪽으로 스크롤하면 근거 열을 모두 볼 수 있습니다.")
    table(diag, height=450)
    with st.expander("명암·그래디언트 프로파일", expanded=True):
        try:
            profile = be.profile_table(dataset, row["category"], row["cd_index"], params=result.params)
            if profile.empty:
                st.info("이미지 또는 유효 좌표가 없어 프로파일을 계산할 수 없습니다.")
            else:
                st.plotly_chart(profile_plot(profile, row), use_container_width=True, config={"displaylogo": False})
                st.json(profile.attrs, expanded=False)
                with st.container(border=True):
                    st.caption("프로파일 모든 샘플 수치")
                    table(profile, height=300)
        except Exception as exc:
            issue_error(exc, "프로파일 표시 실패")
    numeric = [c for c in rows.select_dtypes(include=np.number) if c not in ["source_row", "cd_index"]]
    if numeric:
        feature = st.selectbox("선택 시퀀스의 변화 추이", numeric, index=numeric.index("cd_nm") if "cd_nm" in numeric else 0)
        st.plotly_chart(px.line(rows.sort_values("cd_index"), x="cd_index", y=feature, markers=True,
                                hover_data=["source_row"], labels={"cd_index": "측정 순서"}), use_container_width=True)


def distributions(result):
    st.subheader("D4 · 전체 분포와 카테고리 차이")
    rows = result.l3.copy()
    selected = st.multiselect("분포에 포함할 카테고리", rows.category.astype(str).unique().tolist(),
                              default=rows.category.astype(str).unique().tolist())
    rows = rows[rows.category.astype(str).isin(selected)]
    image_ids = st.multiselect("비교할 이미지 · 비우면 전체", rows.image_id.astype(str).unique().tolist(), key="distribution_images")
    if image_ids:
        rows = rows[rows.image_id.astype(str).isin(image_ids)]
    cols = rows.select_dtypes(include=np.number).columns.tolist()
    if not cols:
        return
    a, b = st.columns(2)
    y = a.selectbox("분포 / 산점도 Y 특징", cols, index=cols.index("z_max") if "z_max" in cols else 0)
    x = b.selectbox("산점도 X 특징", cols, index=cols.index("cd_nm") if "cd_nm" in cols else 0)
    st.caption(f"선택 행 {len(rows)} · {y} 미산출 {rows[y].isna().sum()} · 그래프는 유효 수치만 표시합니다.")
    st.markdown("**카테고리 → 이미지별 비교**")
    if not rows.empty:
        summary = rows.groupby(["category", "image_id"], sort=False, dropna=False).agg(
            n_cd=(y, "size"), feature_median=(y, "median"), feature_missing=(y, lambda v: int(v.isna().sum())),
            z_max=("z_max", "max")).reset_index()
        table(summary, height=240)
        left, middle, right = st.columns([2, 2, 1])
        drill_category = left.selectbox("자세히 볼 카테고리", rows.category.astype(str).unique().tolist(), key="drill_category")
        drill_images = rows.loc[rows.category.astype(str) == drill_category, "image_id"].astype(str).unique().tolist()
        drill_image = middle.selectbox("카테고리 안에서 이미지 선택", drill_images, key="drill_image")
        if right.button("D2에 선택 적용"):
            st.session_state.pending_drilldown = (drill_image, drill_category)
            st.rerun()
        st.caption("‘D2에 선택 적용’ 후 D2 탭에서 해당 이미지의 CD를 고르세요. D3 근거도 같은 CD를 따라갑니다.")
    group_label = st.selectbox("히스토그램 / 산점도 비교 그룹", ["카테고리별", "이미지별", "정상 기준 / 평가 이미지"])
    group_column = {"카테고리별": "category", "이미지별": "image_id", "정상 기준 / 평가 이미지": "reference_group"}[group_label]
    rows["reference_group"] = np.where(rows.image_id.isin(result.baseline_ids), "정상 기준", "평가 이미지")
    normalize = st.checkbox("그룹별 비율로 비교 · 표본 수 차이 보정")
    values = pd.to_numeric(rows[y], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        st.info("선택 특징에 유효 수치가 없습니다. 다른 특징 또는 비교 대상을 고르세요.")
        return
    low, high = float(finite.min()), float(finite.max())
    width = (high-low)/35 if high > low else max(abs(low)*.02, .1)
    bins = dict(start=low-width*.5, end=high+width*.5, size=width)
    hist = go.Figure()
    for name, group in rows.groupby(group_column, sort=False, dropna=False):
        hist.add_trace(go.Histogram(x=group[y], name=str(name), xbins=bins, opacity=.55,
                                   histnorm="probability" if normalize else None))
    hist.update_layout(barmode="overlay", xaxis_title=y, yaxis_title="그룹 내 비율" if normalize else "CD 수", height=380)
    st.plotly_chart(hist, use_container_width=True, key="comparison_hist")
    st.caption("모든 그룹은 같은 구간 경계를 사용합니다. 아래 산점도의 점 또는 값 목록의 행을 클릭하면 바로 실제 이미지 영역이 나타납니다.")
    rows["__source_index"] = rows.index
    scatter = px.scatter(rows, x=x, y=y, color=group_column, hover_data=["image_id", "category", "source_row", "cd_index"],
                         custom_data=["__source_index"], render_mode="svg")
    scatter.update_layout(clickmode="event+select")
    event = st.plotly_chart(scatter, use_container_width=True, on_select=lambda: st.session_state.update(distribution_selection_kind="scatter"),
                            selection_mode="points", key=f"value_scatter_{x}_{y}_{group_column}_{hash(tuple(rows.index))}")
    ranges = st.columns(2)
    lower = ranges[0].number_input("값 목록 최소값 (Y 특징)", value=low, key=f"value_min_{y}_{hash(tuple(rows.index))}")
    upper = ranges[1].number_input("값 목록 최대값 (Y 특징)", value=high, key=f"value_max_{y}_{hash(tuple(rows.index))}")
    candidates = rows[values.between(lower, upper)]
    cols_to_show = list(dict.fromkeys(["image_id", "category", "source_row", "cd_index", y, x, "z_max"]))
    selection = st.dataframe(candidates[cols_to_show], use_container_width=True, height=260, hide_index=True,
                              on_select=lambda: st.session_state.update(distribution_selection_kind="table"),
                              selection_mode="single-row", key=f"value_rows_{y}_{hash(tuple(candidates.index))}")
    clicked = selected_source_index(candidates, selection["selection"]["rows"])
    points = event["selection"]["points"]
    if st.session_state.get("distribution_selection_kind") == "scatter" and points and points[-1].get("customdata"):
        point_index = points[-1]["customdata"][0]
        if point_index in rows.index:
            clicked = point_index
    if clicked is not None:
        linked_preview(result, clicked, "distribution")
    else:
        st.info("산점도의 점 또는 값 목록의 행을 클릭해 실제 이미지의 해당 영역을 확인하세요.")


def groups(result):
    st.subheader("D5 · 시퀀스 전체 / 이미지 전체 이상")
    st.caption("L2는 이미지 × 카테고리의 길이·피치·각도·궤적 요약, L1은 이미지 전체 명암·노이즈·대비입니다. 정상 기준 이미지 수가 적으면 점수가 미산출될 수 있습니다.")
    image_filter = st.multiselect("L2 / L1에 표시할 이미지 · 비우면 전체", [d.image_id for d in result.datasets])
    a, b = st.tabs(["L2 시퀀스", "L1 이미지"])
    for tab, level in [(a, "l2"), (b, "l1")]:
        with tab:
            data = getattr(result, level)
            if image_filter and "image_id" in data:
                data = data[data.image_id.isin(image_filter)]
            table(data, height=320)
            if not data.empty:
                idx = st.selectbox(f"{level.upper()} 상세 행", data.index.tolist(),
                                    format_func=lambda i, data=data: " · ".join(str(data.loc[i, c]) for c in ["image_id", "category"] if c in data), key="detail_" + level)
                table(be.feature_diagnostics(result, idx, level=level), height=300)


def calibration(result):
    st.subheader("D6 · 기준 통계가 믿을 만한지 확인하기")
    st.write("사용 기준:", st.session_state.get("run_mode", ""))
    st.write("정상 기준 이미지 ID:", result.baseline_ids)
    st.caption("표본 수 n, 원래 MAD, 적용 하한과 최종 scale, 로그 변환 여부를 함께 보세요. 기준이 좁거나 부족한 경우 작은 차이도 큰 점수 또는 NaN이 됩니다.")
    records = []
    for level in ("l3", "l2", "l1"):
        for category, stats in result.stats.get(level, {}).items():
            for feature, values in stats.get("features", {}).items():
                mad = values.get("mad", np.nan)
                raw_mad = stats.get("raw_features", {}).get(feature, {}).get("mad", np.nan)
                records.append(dict(level=level, category=category, feature=feature,
                    n=stats.get("n", 0), valid_n=stats.get("valid_counts", {}).get(feature, 0),
                    median=values.get("median", np.nan), raw_mad=raw_mad, floored_mad=mad,
                    scale=max(1.4826*mad, result.params.mad_floor(feature)) if np.isfinite(mad) else np.nan,
                    space="log(x+0.001)" if feature in result.params.log_features else "raw"))
    table(records, height=440)
    with st.expander("기준 통계 원본 JSON · 버전 / 최빈값 포함"):
        st.json(result.stats, expanded=False)
    st.download_button("현재 기준 통계 JSON 저장", be.export_stats(result), "cdqc-reference.json", "application/json")
    with st.expander("모든 특징의 정의 · 방향 · 사유 코드"):
        table(be.registry_table(), height=400)


def feedback(result):
    st.subheader("D7 · 관찰 기록과 다음 대화 준비")
    st.caption("화면 번호와 관찰을 말로 전달할 수 있습니다. 아래 메모에는 실제 이미지 ID나 경로를 자동으로 넣지 않습니다. 작성한 내용은 이 PC에서만 내려받습니다.")
    section = st.selectbox("어느 화면에서 관찰했나요?", ["D0 파일 연결", "D1 전체 요약", "D2 좌표/이미지", "D3 개별 특징/프로파일", "D4 분포", "D5 시퀀스/이미지", "D6 기준 통계"])
    prompts = ["정상으로 보이는데 점수가 큼", "이상으로 보이는데 점수가 낮음", "좌표가 일정하게 밀림", "일부 카테고리에서만 발생", "NaN / 표본 부족이 많음", "상한 점수가 많음", "연결/로딩 문제", "기타"]
    observation = st.multiselect("관찰한 현상", prompts)
    direction = st.selectbox("대략적인 범위", ["확인 전", "거의 전체", "일부 이미지", "특정 카테고리", "소수 CD"])
    note = st.text_area("직접 적는 메모 · 공유 가능한 범위에서 작성", height=160,
                         placeholder="예: D2에서 대부분 S/E가 같은 방향으로 약 1px 밀려 보임. D3 delta는 한쪽 부호로 모임. 정상 기준은 12장.")
    template = f"화면: {section}\n현상: {', '.join(observation)}\n범위: {direction}\n메모: {note}\n"
    st.code(template, language=None)
    st.download_button("관찰 메모 TXT 저장", template, "observation-notes.txt", "text/plain")
    with st.expander("다음 질문에 바로 답하려면"):
        st.markdown("- 좌표는 어느 방향으로 몇 px 밀리나요? → **D2**에서 S/E와 좌표 확인\n- 어떤 특징 때문에 점수가 높나요? → **D3**에서 directed z 정렬, raw / 기준값 비교\n- 이미지가 흐린가요, 좌표만 잘못됐나요? → **D3**에서 명암·피크와 보고 위치 비교\n- 한 카테고리만 다른가요? → **D4** 카테고리 분포\n- CD 전체가 같이 움직이나요? → **D5** 각도·피치·궤적\n- 기준이 너무 좁은가요? → **D6** n / MAD / scale 확인")
    with st.expander("실행 진단 · 입력 문제 · 실제 사용한 설정", expanded=False):
        st.write("실행 ID:", str(result.run_id), " · 이 세션 분석 횟수:", st.session_state.run_count)
        table(result.issues)
        table(st.session_state.get("load_issues", []))
        st.json({"params": asdict(result.params) if isinstance(result.params, Params) else result.params,
                 "config": st.session_state.get("run_config", {}),
                 "scan_config": st.session_state.get("run_scan_config", {}),
                 "effective_manifest": st.session_state.get("run_manifest", [])}, expanded=False)
        if st.session_state.get("failed_attempt"):
            st.warning("아래는 실패한 최근 시도입니다. 위의 성공 결과와 입력 정보는 유지됩니다.")
            st.json(st.session_state.failed_attempt, expanded=False)
        if st.session_state.get("last_error"):
            st.code(st.session_state.last_error)
    with st.expander("로컬 결과 내보내기 · 전체 식별자와 수치 포함"):
        st.caption("아래 CSV와 기준 JSON은 익명화되지 않습니다. 사내 분석·보관용이며 외부 전송 기능은 없습니다.")
        for level in ["l3", "l2", "l1"]:
            st.download_button(f"{level.upper()} 전체 CSV 저장", getattr(result, level).to_csv(index=False).encode("utf-8-sig"),
                                f"cdqc-{level}.csv", "text/csv", key="export_" + level)


def main():
    st.set_page_config(page_title="CDQC · 품질 진단 작업대", page_icon="🔬", layout="wide")
    init_state()
    imported = st.session_state.pop("pending_import_config", None)
    if imported:
        st.session_state.config = imported
        st.session_state.params_json = json_text(imported["params"])
        for key in ["root_path", "pattern_image", "pattern_table", "pattern_seg", "pattern_dm3", "role_rules"]:
            st.session_state.pop(key, None)
    pending = st.session_state.pop("pending_drilldown", None)
    if pending:
        st.session_state.selected_image, st.session_state.selected_category = pending
        st.session_state.pop("selected_cd", None)
    focused = st.session_state.pop("pending_focus", None)
    if focused:
        st.session_state.selected_image, st.session_state.selected_category, st.session_state.selected_cd = focused
    if st.session_state.pop("params_reset", False):
        st.session_state.params_json = json_text(asdict(Params()))
    st.markdown("<style>.block-container{padding-top:2rem;max-width:1600px} div[data-testid='stMetric']{background:#eef6f5;border-radius:12px;padding:16px} h1{letter-spacing:-.04em} [data-testid='stSidebar']{background:#edf3f4}</style>", unsafe_allow_html=True)
    st.title("CDQC · 품질 진단 작업대")
    st.markdown("**폴더부터 CD 한 개의 근거까지.** 사내 데이터를 보면서 확인하고, 화면 번호와 관찰 내용으로 개선을 이어가세요.")
    with st.sidebar:
        st.markdown("### 작업 순서")
        st.markdown("**1** 폴더 · 규칙 확인\n\n**2** 연결 미리보기\n\n**3** 정상 기준 선택 후 분석\n\n**4** D1–D7에서 원인 확인")
        demo = st.button("합성 데모 바로 분석", use_container_width=True, type="primary", key="demo_run")
        st.caption("원본 파일 없이 작동을 확인합니다. 실제 성능 검증용 데이터는 아닙니다.")
        if st.session_state.analysis is not None:
            st.success(f"분석 결과 유지 중 · {st.session_state.run_count}회 실행")
            snapshot = st.session_state.analysis
            if all(d.metadata.get("demo") for d in snapshot.datasets):
                st.warning("현재 표시 결과: 합성 데모")
            else:
                st.markdown("**현재 표시 결과: 실제 파일 분석**")
                st.caption(str(st.session_state.get("run_config", {}).get("root", "")))
            st.caption(f"실행 ID: {snapshot.run_id}")
        st.info("화면 필터·확대·정렬은 분석을 다시 실행하지 않습니다. 설정 변경은 분석 버튼을 누른 다음 반영됩니다.")
    cfg, auto = configure()
    matches = matches_preview()
    ids = matches.image_id.astype(str).tolist() if not matches.empty else [d.image_id for d in st.session_state.datasets]
    mode, chosen = analysis_settings(ids)
    scan_keys = ("root", "recursive", "patterns", "match_mode", "roles", "key_source", "key_regex", "role_key_regex", "require_seg", "require_dm3", "px_nm", "ignored_patterns")
    scanned = st.session_state.get("scanned_config", {})
    dirty = cfg is not None and not matches.empty and any(cfg.get(k) != scanned.get(k) for k in scan_keys)
    if dirty:
        st.warning("파일 연결 규칙 또는 공통 픽셀 크기가 탐색 이후 바뀌었습니다. 다시 탐색한 뒤 분석하세요. 좌표 보정 옵션만 바꾼 경우에는 재탐색이 필요 없습니다.")
    demo_ready = bool(st.session_state.datasets) and all(d.metadata.get("demo") for d in st.session_state.datasets) and not st.session_state.get("scan_attempted")
    run = st.button("선택 묶음 분석 / 다시 분석", disabled=cfg is None or dirty or (matches.empty and not demo_ready), key="analyze_run")
    if demo:
        run_analysis(cfg or ds.default_config(), pd.DataFrame(), "탐색: 입력 전체", [], demo=True)
    elif (run or auto) and cfg is not None:
        run_analysis(cfg, matches, mode, chosen)
    result = st.session_state.analysis
    if result is None:
        st.info("처음이라면 왼쪽 ‘합성 데모 바로 분석’으로 모든 화면을 먼저 확인하세요. 실제 데이터는 D0에서 루트 폴더 하나를 지정합니다.")
        if st.session_state.get("load_issues"):
            table(st.session_state.load_issues)
        if st.session_state.get("failed_attempt"):
            st.markdown("**실패한 분석 시도 · 입력 및 로딩 진단**")
            st.json(st.session_state.failed_attempt, expanded=False)
        return
    if result.l3.empty:
        st.warning("분석 결과에 CD 행이 없습니다.")
        return
    st.divider()
    tabs = st.tabs(["D1 전체 요약", "D2 이미지 · CD", "D3 점수 근거", "D4 분포", "D5 L2 · L1", "D6 기준 통계", "D7 관찰 기록"])
    with tabs[0]:
        overview(result)
    with tabs[1]:
        dataset, row, index, rows = select_cd(result)
    with tabs[2]:
        diagnostics(result, dataset, row, index, rows)
    with tabs[3]:
        distributions(result)
    with tabs[4]:
        groups(result)
    with tabs[5]:
        calibration(result)
    with tabs[6]:
        feedback(result)


if __name__ == "__main__":
    main()
