"""피쳐 레지스트리 — 단일 목록 (spec §8.2).

피쳐 추가 = 여기 한 줄 + 계산 함수 하나. 이 목록이 정규화 방향(worse_when),
사유 코드, 리포트, selftest 기대 표를 전부 구동한다.

kind:
    z        코호트 robust z로 정규화되는 연속 피쳐
    raw      계산·기록만 하고 정규화하지 않음 (z가 수학적으로 성립하지 않는
             원형 변수 등. 파생 피쳐의 입력으로 남긴다)
    bool     유효성 플래그. False면 고정 z(Z_ON_BAD) 부여
    match    범주형. 코호트 최빈값과 불일치하면 고정 z(Z_ON_BAD) 부여
    physical 물리 단위(nm) 그대로 공차와 비교 (impact_nm)
    rollup   다른 레벨 결과의 집계 (정규화 안 함)

worse_when (kind="z"의 정규화 방향, 그리고 kind와 무관하게 abs_flags의 비교 방향):
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
    # σ 스윕 delta (spec §3.2) — grad_sigma_px 의존성을 없앤 delta와 그 산포.
    # 단위는 이름대로 **px** (delta_s/delta_e는 nm)
    _F("delta_B_s", "l3", "z", "both", "σ 스윕 S delta의 중앙값(px) — grad_sigma_px 비의존 delta", "POSITION_MISMATCH"),
    _F("delta_B_e", "l3", "z", "both", "σ 스윕 E delta의 중앙값(px)", "POSITION_MISMATCH"),
    _F("delta_scatter_s", "l3", "z", "high", "σ 스윕 S delta의 MAD(px) — 크면 엣지 자체가 없어 피크가 σ 따라 떠다님", "EVIDENCE_WEAK"),
    _F("delta_scatter_e", "l3", "z", "high", "σ 스윕 E delta의 MAD(px)", "EVIDENCE_WEAK"),
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
    # angle은 z 대상이 아니다 — 180° 주기 원형 변수라 뺄셈이 정의되지 않고
    # (kind="raw": 계산·기록만), 정규화는 원형 잔차 angle_resid로 한다
    _F("angle", "l3", "raw", "-", "세그먼트 절대 각도(deg). L2 angle_median의 입력 — z 대상 아님"),
    _F("angle_resid_seq", "l3", "z", "both", "**시퀀스** 원형 중앙값 대비 각도 잔차(deg, 180° 주기) — 이웃과 방향이 다른 CD. 시퀀스가 통째로 돌면 0", "GEOMETRY_ODD"),
    # 코호트 중심 잔차는 cohort_z가 붙인다 (코호트 전체를 봐야 계산됨).
    # 시퀀스 전체 회전이 여기서는 잡히고 angle_resid_seq에서는 안 잡힌다 —
    # 어느 쪽이 유용한지는 데이터가 쌓여야 알므로 둘 다 계산한다
    _F("angle_resid_cohort", "l3", "raw", "both", "**코호트** 원형 중앙값 대비 각도 잔차(deg) — 시퀀스 전체 회전도 여기서는 걸린다. cohort_z가 계산"),
    _F("value_mismatch_nm", "l3", "z", "both", "보고 측정값과 좌표 기하 길이의 차(nm) — 좌표·값 불일치 감지", "GEOMETRY_ODD"),
    # ---- L3 마스크 정합 (extract_mask_l3 — 마스크가 있는 카테고리만) --------
    _F("mdist_s", "l3", "z", "high", "보고 S좌표 ↔ 가장 가까운 마스크 경계 거리(nm)", "POSITION_MISMATCH"),
    _F("mdist_e", "l3", "z", "high", "보고 E좌표 ↔ 가장 가까운 마스크 경계 거리(nm)", "POSITION_MISMATCH"),
    _F("mgrad_s", "l3", "z", "low", "S 최근접 마스크 경계점의 이미지 그래디언트/국소 σ — DL 경계가 실제 전이 위인가", "POSITION_MISMATCH"),
    _F("mgrad_e", "l3", "z", "low", "E 최근접 마스크 경계점의 이미지 그래디언트/국소 σ", "POSITION_MISMATCH"),
    _F("minside", "l3", "bool", "-", "세그먼트 중점이 마스크 내부 — CD가 마스크 구조를 실제로 가로지르는가", "GEOMETRY_ODD"),
    # ---- L3 라벨 경계 정합 (boundary_features — **권장 경로**) --------------
    #      라벨맵의 *모든* 전이의 합집합이 경계다. 클래스를 고를 필요가 없어
    #      S와 E가 서로 다른 계면에 있어도 측정된다 (mdist_*는 클래스 하나만 봄)
    _F("bdist_s", "l3", "z", "high", "보고 S좌표 ↔ 가장 가까운 라벨 전이 거리(nm). 정상 계면 측정이면 ≈0", "POSITION_MISMATCH"),
    _F("bdist_e", "l3", "z", "high", "보고 E좌표 ↔ 가장 가까운 라벨 전이 거리(nm)", "POSITION_MISMATCH"),
    _F("bdist_mid", "l3", "z", "both", "세그먼트 중점 ↔ 라벨 전이 거리(nm) — bdist_ratio의 분모(층 반폭 대리)", "GEOMETRY_ODD"),
    _F("bdist_ratio", "l3", "z", "high", "max(bdist_s, bdist_e) / bdist_mid — 배율 무관. 정상 CD ≈0, 허공에 그은 선 ≈1", "POSITION_MISMATCH"),
    # 방향은 "both": 예상보다 많이 지나도(다중 관통) 적게 지나도(허공에 그은 선)
    # 이상이다. 카테고리별 z이므로 그 카테고리의 정상 통과 횟수가 중앙값이 된다
    _F("label_runs", "l3", "z", "both", "S→E 안쪽 구간에서 만나는 라벨 런 개수 — 1이면 한 층 안에만 있음(계면을 안 가로지름)", "GEOMETRY_ODD"),
    _F("bgrad_s", "l3", "z", "low", "보고 S좌표 **그 자리**의 이미지 그래디언트/국소 σ — 좌표에 실제 명암 전이가 있는가 (프로파일 기반 cnr과 독립)", "POSITION_MISMATCH"),
    _F("bgrad_e", "l3", "z", "low", "보고 E좌표 그 자리의 이미지 그래디언트/국소 σ", "POSITION_MISMATCH"),
    _F("label_s", "l3", "raw", "-", "S에서 안쪽으로 label_inset_px 지점의 라벨값 — 이 CD가 어느 층을 재는가"),
    _F("label_e", "l3", "raw", "-", "E에서 안쪽으로 label_inset_px 지점의 라벨값"),
    _F("label_mid", "l3", "raw", "-", "세그먼트 중점의 라벨값"),
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
    _F("bdist_median_s", "l2", "z", "high", "S 라벨 경계 거리 중앙값(nm) — 시퀀스 전체가 경계에서 이탈", g2=True),
    _F("bdist_median_e", "l2", "z", "high", "E 라벨 경계 거리 중앙값(nm)", g2=True),
    # ---- L2 총체적 실패 (시퀀스 전체가 다른 모양 — 변경 #03 §2) -------------
    _F("angle_median", "l2", "z", "both", "세그먼트 각도의 원형 중앙값(deg, 180° 주기) — 기준 각도 오설정. 코호트가 ±90° 랩 경계 근처면 주의", g2=True),
    _F("angle_spread", "l2", "z", "high", "각도의 원형 MAD(deg) — 방향이 뒤죽박죽인 시퀀스", g2=True),
    _F("pitch_median", "l2", "z", "both", "이웃 CD 중점 간 거리 중앙값(nm) — 측정 간격 설정 오류", g2=True),
    _F("span_nm", "l2", "z", "both", "첫/끝 CD 중점 간 거리(nm) — ROI 길이가 다름", g2=True),
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
    # ---- LM 마스크 이미지 레벨 (extract_mask_image — 이미지×클래스당 하나.
    #      L1과 코호트 축이 다를 수 있음. 코호트 분리는 사용자 몫) ------------
    _F("mask_grad_agree", "lm", "z", "low", "마스크 경계 전체의 이미지 그래디언트 중앙값/노이즈 σ — DL이 헛것을 그렸는지의 단일 지표"),
    _F("mask_n_components", "lm", "z", "both", "연결 성분 수 (8-이웃)"),
    _F("mask_hole_frac", "lm", "z", "high", "성분 내부 구멍 픽셀 비율"),
    _F("mask_boundary_rough", "lm", "z", "high", "경계 둘레/등면적 원 둘레 (16px 이상 성분별 중앙값) — 매끄러움 대리"),
    _F("mask_area_frac", "lm", "z", "both", "마스크 픽셀 비율"),
]

BY_NAME: dict[str, Feature] = {f.name: f for f in REGISTRY}


def features_of(level: str, kind: str | None = None) -> list[Feature]:
    return [f for f in REGISTRY if f.level == level and (kind is None or f.kind == kind)]


