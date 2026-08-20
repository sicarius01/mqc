"""피쳐 레지스트리 — 단일 목록 (spec §8.2).

피쳐 추가 = 여기 한 줄 + 계산 함수 하나. 이 목록이 정규화 방향(worse_when),
사유 코드, 리포트, selftest 기대 표를 전부 구동한다.

kind:
    z        코호트 robust z로 정규화되는 연속 피쳐
    bool     유효성 플래그. False면 고정 z(Z_ON_BAD) 부여
    match    범주형. 코호트 최빈값과 불일치하면 고정 z(Z_ON_BAD) 부여
    physical 물리 단위(nm) 그대로 공차와 비교 (impact_nm)
    rollup   다른 레벨 결과의 집계 (정규화 안 함)

worse_when (kind="z"에만 의미):
    low   낮을수록 나쁨 → directed z = -z
    high  높을수록 나쁨 → directed z = +z
    both  양쪽 다 → directed z = |z|
"""

from __future__ import annotations

from dataclasses import dataclass, field

# bool/match 피쳐가 나쁠 때 부여하는 고정 directed z (t_soft 위로 확실히 올라가게)
Z_ON_BAD = 4.0

# CD 레벨 사유 코드 (spec §4.3)
REASONS = ("EVIDENCE_WEAK", "POSITION_MISMATCH", "SEQUENCE_JUMP",
           "GEOMETRY_ODD", "FOCUS_BLUR")


@dataclass(frozen=True)
class Feature:
    name: str
    level: str            # "l3" | "l2" | "l1"
    kind: str             # "z" | "bool" | "match" | "physical" | "rollup"
    worse_when: str       # "low" | "high" | "both" | "-"
    desc: str             # 피쳐의 의미 한 줄
    reason: str = ""      # L3 플래그 시 사유 코드
    computed_at: str = "extract"   # "extract" | "run" (플래그 의존 피쳐는 run)
    g2: bool = False      # L2 중 G2 게이트(시퀀스 요약 z) 입력 여부
    g0: bool = False      # L1 중 G0 게이트 입력 여부
    enabled_default: bool = True   # enabled_* = "all"일 때 포함 여부
                                   # (계산·리포트는 되지만 플래그를 구동하지 않음)


_F = Feature

REGISTRY: list[Feature] = [
    # ---- L3 이미지 증거 (엔드포인트별) --------------------------------------
    _F("delta_s", "l3", "z", "both", "보고 S좌표와 그래디언트 피크의 부호 있는 거리(nm). 1순위 증거", "POSITION_MISMATCH"),
    _F("delta_e", "l3", "z", "both", "보고 E좌표와 그래디언트 피크의 부호 있는 거리(nm)", "POSITION_MISMATCH"),
    _F("cnr_s", "l3", "z", "low", "S 스텝 높이 / 국소 노이즈 σ — 엣지 증거 강도", "EVIDENCE_WEAK"),
    _F("cnr_e", "l3", "z", "low", "E 스텝 높이 / 국소 노이즈 σ", "EVIDENCE_WEAK"),
    _F("rise_s", "l3", "z", "high", "S 엣지 10–90% 상승폭(nm) — 클수록 흐림", "FOCUS_BLUR"),
    _F("rise_e", "l3", "z", "high", "E 엣지 10–90% 상승폭(nm)", "FOCUS_BLUR"),
    _F("margin_s", "l3", "z", "low", "S 그래디언트 1등 피크/2등 피크 비 — 낮으면 경쟁 엣지 존재", "EVIDENCE_WEAK"),
    _F("margin_e", "l3", "z", "low", "E 그래디언트 1등/2등 피크 비", "EVIDENCE_WEAK"),
    _F("npk_s", "l3", "z", "high", "S 창 내 유의 그래디언트 피크 개수 — 다중 엣지 감지", "EVIDENCE_WEAK"),
    _F("npk_e", "l3", "z", "high", "E 창 내 유의 그래디언트 피크 개수", "EVIDENCE_WEAK"),
    _F("overshoot_s", "l3", "z", "high", "S 스텝 양쪽 오버슈트/스텝 높이 — 프레넬 프린지 대리", "FOCUS_BLUR"),
    _F("overshoot_e", "l3", "z", "high", "E 스텝 양쪽 오버슈트/스텝 높이", "FOCUS_BLUR"),
    _F("plateau_cv", "l3", "z", "high", "S–E 플래토 robust 변동계수 — 플래토 결함 감지", "EVIDENCE_WEAK"),
    _F("pol_s", "l3", "match", "-", "S 스텝 극성(±1) — 코호트 최빈값과 불일치면 옆 엣지 락온 의심", "EVIDENCE_WEAK"),
    _F("pol_e", "l3", "match", "-", "E 스텝 극성(±1)", "EVIDENCE_WEAK"),
    _F("edge_valid_s", "l3", "bool", "-", "S 피크가 탐색 창 경계에 걸리지 않음", "EVIDENCE_WEAK"),
    _F("edge_valid_e", "l3", "bool", "-", "E 피크가 탐색 창 경계에 걸리지 않음", "EVIDENCE_WEAK"),
    # ---- L3 기하/시퀀스 잔차 (전부 국소 추세 대비) ---------------------------
    _F("cd_nm", "l3", "z", "both", "CD 길이(nm) — 코호트 대비", "GEOMETRY_ODD"),
    _F("cd_resid", "l3", "z", "both", "국소 robust 추세 대비 CD 잔차(nm)", "SEQUENCE_JUMP"),
    _F("s_resid", "l3", "z", "both", "S 궤적 국소 적합 대비 법선 방향 잔차(nm)", "SEQUENCE_JUMP"),
    _F("e_resid", "l3", "z", "both", "E 궤적 국소 적합 대비 법선 방향 잔차(nm)", "SEQUENCE_JUMP"),
    _F("dstep_s", "l3", "z", "high", "S 이웃 간 1차 차분 크기(nm) — 점프 감지", "SEQUENCE_JUMP"),
    _F("dstep_e", "l3", "z", "high", "E 이웃 간 1차 차분 크기(nm)", "SEQUENCE_JUMP"),
    _F("obliquity", "l3", "z", "high", "세그먼트-엣지접선 각의 시퀀스 중앙값 대비 편차(deg)", "GEOMETRY_ODD"),
    # curv는 노이즈와 진짜 곡률을 구분 못 해 기본 비활성 (리포트용) — s_resid가 대체
    _F("curv_s", "l3", "z", "high", "S 궤적 3점 국소 곡률(2차 차분, nm)", "SEQUENCE_JUMP", enabled_default=False),
    _F("curv_e", "l3", "z", "high", "E 궤적 3점 국소 곡률(nm)", "SEQUENCE_JUMP", enabled_default=False),
    _F("angle", "l3", "z", "both", "세그먼트 절대 각도(deg) — 코호트 대비", "GEOMETRY_ODD"),
    _F("value_mismatch_nm", "l3", "z", "both", "보고 측정값과 좌표 기하 길이의 차(nm) — 좌표·값 불일치 감지", "GEOMETRY_ODD"),
    # ---- L2 카테고리 시퀀스 (image × category) ------------------------------
    _F("n_cd", "l2", "z", "both", "CD 개수 — 코호트 최빈값 대비 (missing 감지)", g2=True),
    _F("frac_flagged", "l2", "z", "high", "L3 플래그 비율", computed_at="run"),
    _F("max_run", "l2", "rollup", "-", "연속 플래그 최대 길이 — 사유 코드 세분화용 (게이트 아님)", computed_at="run"),
    _F("cd_median", "l2", "z", "both", "시퀀스 CD 중앙값(nm) — 코호트 대비", g2=True),
    _F("cd_mad", "l2", "z", "high", "시퀀스 CD MAD(nm) — 산포 이상", ),
    _F("delta_median_s", "l2", "z", "both", "S delta 시퀀스 중앙값(nm) — DL 계통 편향 감지", g2=True),
    _F("delta_median_e", "l2", "z", "both", "E delta 시퀀스 중앙값(nm)", g2=True),
    _F("traj_rms_s", "l2", "z", "high", "S 궤적 잔차 RMS(nm)"),
    _F("traj_rms_e", "l2", "z", "high", "E 궤적 잔차 RMS(nm)"),
    _F("impact_nm", "l2", "physical", "high", "플래그 CD 제외 시 보고 통계량 변화(nm) — G1, 공차와 직접 비교", computed_at="run"),
    # ---- L1 이미지 (측정 무관 OOD) ------------------------------------------
    _F("noise_sigma", "l1", "z", "high", "Laplacian 고주파 잔차 MAD×1.4826 — 노이즈 수준", g0=True),
    _F("sat_lo", "l1", "z", "high", "값 0 픽셀 비율 — 하위 새추레이션", g0=True),
    _F("sat_hi", "l1", "z", "high", "값 255 픽셀 비율 — 상위 새추레이션", g0=True),
    _F("dyn_range", "l1", "z", "low", "p99 − p1 — 다이나믹 레인지", g0=True),
    _F("hist_emd", "l1", "z", "high", "코호트 중앙값 히스토그램과의 EMD — 분포 이탈", g0=True, computed_at="run"),
    _F("struct_energy", "l1", "z", "low", "그래디언트 p90 / noise_sigma — 구조 대비 노이즈", g0=True),
    _F("tile_energy_cv", "l1", "z", "high", "타일별 struct_energy CV — 부분 손상", g0=True),
    _F("n_bad_categories", "l1", "rollup", "-", "이미지 내 L2 게이트 발동 카테고리 수 (롤업)", computed_at="run"),
]

BY_NAME: dict[str, Feature] = {f.name: f for f in REGISTRY}


def features_of(level: str, kind: str | None = None) -> list[Feature]:
    return [f for f in REGISTRY if f.level == level and (kind is None or f.kind == kind)]


