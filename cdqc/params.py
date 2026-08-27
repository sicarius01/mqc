"""연산 파라미터 — config.toml을 대체하는 단일 dataclass.

모든 공개 API 함수가 Params 하나를 받는다. 기본값은 여기 내장돼 있고,
사용자는 필드를 바꾼 인스턴스를 만들어 쓰면 된다. 저장/로드가 필요하면
dataclasses.asdict()로 사용자가 알아서 한다 — cdqc는 파일을 만지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_mad_floors() -> dict[str, float]:
    """이산/퇴화 피쳐의 MAD=0 방어 (spec §4.2). 단위: 피쳐 각자의 단위.

    log_features에 속한 피쳐의 하한은 **로그 공간** 값이다 (cohort_stats가
    변환 후 MAD를 재므로) — 대략 상대 변화율로 읽으면 된다.
    """
    return {
        "npk_s": 0.5, "npk_e": 0.5,
        "label_runs": 0.5,                          # 정수 (라벨 전이 개수)
        "n_cd": 0.5,
        "sat_lo": 0.002, "sat_hi": 0.002,          # 픽셀 비율
        "hist_emd": 0.5,
        "frac_flagged": 0.05,
        "overshoot_s": 0.1, "overshoot_e": 0.1,     # log 공간 (≈10% 상대)
        "max_run": 0.5,
        "noise_sigma": 0.05,                        # gray level (uint8 양자화 방어)
        "dyn_range": 1.0,
        "delta_median_s": 0.05, "delta_median_e": 0.05,   # nm (subpixel 분해능)
        # L2 요약값 — 레시피가 규칙적이면 카테고리 내 변동이 구조적으로 0이다
        # (피치가 일정, 길이가 일정). 물리 분해능 수준의 하한을 깔아둔다
        "cd_median": 0.02, "pitch_median": 0.05, "span_nm": 0.05,   # nm
        "traj_rms_s": 0.02, "traj_rms_e": 0.02,     # nm
        "angle_median": 0.05,                        # deg
        "value_mismatch_nm": 0.02,                  # nm
        "mdist_s": 0.05, "mdist_e": 0.05,           # nm (마스크 경계 픽셀 양자화)
        "mask_n_components": 0.5,                   # 정수
        "mask_hole_frac": 0.002,
        "mask_area_frac": 0.002,
        "angle_spread": 0.05,                       # deg
        # 라벨 경계 거리 — 정상 CD는 정확히 0.000이라 카테고리 MAD가 구조적으로
        # 죽는다 (§8-1). 물리 절대 임계와 병행해서 쓰는 것을 권장
        "bdist_s": 0.05, "bdist_e": 0.05, "bdist_mid": 0.05,   # nm
        "bdist_ratio": 0.02,                        # 무차원 비
        "bdist_median_s": 0.05, "bdist_median_e": 0.05,        # nm (L2)
        # σ 스윕 delta — subpixel 분해능 수준
        "delta_B_s": 0.02, "delta_B_e": 0.02,       # px
        "delta_scatter_s": 0.05, "delta_scatter_e": 0.05,      # log 공간
    }


@dataclass
class Params:
    """cdqc 연산 파라미터. 필드 의미는 cdqc_spec.md §3, §4 참조."""

    # ---- 프로파일 샘플링 (spec §3.2) ------------------------------------
    ribbon_half_w: int = 2        # 엣지 접선 방향 ±px 리본 평균
    win_frac: float = 0.4         # 탐색 창 = win_frac × CD
    win_min_px: float = 8.0
    win_max_px: float = 64.0
    margin_px: float = 6.0        # S/E 바깥 여유 하한 (유효값은 max(margin, W/2+2))
    grad_sigma_px: float = 1.0    # 그래디언트용 가우시안
    noise_patch_px: int = 15      # 국소 노이즈 추정 패치
    peak_suppress: int = 6        # 2등 피크 탐색 시 1등 주변 억제 폭 (샘플)
    npk_ratio: float = 0.3        # 유의 피크 기준 (1등 피크 대비)

    # ---- delta σ 스윕 (spec §3.2) ---------------------------------------
    # grad_sigma_px 하나에 delta가 의존하지 않도록 σ를 훑어 중앙값(delta_B)과
    # MAD(delta_scatter)를 낸다. scatter는 그 자체가 피쳐다:
    #   delta 큼 + scatter 작음 → 엣지는 선명한데 좌표가 딴 데 (측정 실패)
    #   delta 큼 + scatter 큼   → 엣지 자체가 없어 아무 데나 (이미지 문제)
    delta_sigma_sweep: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

    # ---- 라벨 경계 (boundary_features) ------------------------------------
    label_inset_px: float = 2.0   # label_s/e 샘플 지점을 끝점에서 안쪽으로 이만큼.
                                  # 끝점은 계면 위라 라벨이 양쪽을 오간다 —
                                  # label_runs도 이 구간을 뺀 안쪽에서 센다

    # ---- 비율 피쳐 수치 안정성 (sanitize, spec §3.2) ----------------------
    # **카테고리 예외가 아니라 피쳐 자체의 일반 규칙이다** — 스텝이 노이즈
    # 수준이면 비율 피쳐의 분모가 0에 가까워 물리적으로 불가능한 값이 나온다.
    min_cnr_for_ratio: float = 1.0   # cnr < 이것이면 overshoot → NaN
    min_rise_px: float = 0.5         # rise < 이것(px)이면 rise/overshoot → NaN

    # ---- 시퀀스 잔차 ------------------------------------------------------
    local_window: int = 9         # 국소 추세 창 (홀수). 구조 변화 스케일보다 짧게
    method: str = "robust_linear"  # robust_linear(국소 Theil-Sen) | hampel(러닝 메디안)
    min_seq_len: int = 5          # 미만이면 시퀀스 잔차 NaN

    # ---- 코호트 통계 / z --------------------------------------------------
    trim_frac: float = 0.05       # 오염 방어 트림 (1차 z 상위 이 비율 제거 후 재계산)
    mad_floor_default: float = 1e-6
    mad_floors: dict[str, float] = field(default_factory=_default_mad_floors)
    # 양수 heavy-tail 피쳐 — log(x + 1e-3) 후 z (spec §4.2)
    log_features: tuple[str, ...] = ("rise_s", "rise_e", "margin_s", "margin_e",
                                     "dstep_s", "dstep_e", "cd_mad",
                                     "overshoot_s", "overshoot_e",
                                     "delta_scatter_s", "delta_scatter_e")
    direction_overrides: dict[str, str] = field(default_factory=dict)
    # ^ 피쳐별 worse_when 오버라이드 ("low"|"high"|"both"). 기본은 registry
    min_cohort_n: int = 5
    # ^ 코호트 안에서 유효값이 이보다 적은 피쳐는 통계를 내지 않는다 (NaN).
    #   n=2로 낸 MAD는 의미가 없다 — 억지로 z를 만드느니 n_z_valid에 반영한다
    rel_scale_floor: float = 0.01
    # ^ 분모의 **상대** 하한: 0.01·|중앙값|. 중심이 큰 피쳐(cd_nm이 수십 nm)에서
    #   MAD만 작으면 z가 폭발한다
    abs_scale_floor: float = 1e-3   # 분모의 절대 바닥
    pooled_mad_frac: float = 0.30
    # ^ 코호트(카테고리) MAD가 죽었을 때 전체 pooled MAD의 이 비율을 분모 하한으로
    #   (spec §4.2 — 피쳐를 빼서가 아니라 분모 하한으로 푸는 문제)
    z_cap: float = 30.0
    # ^ directed z 상한. MAD≈0 잔여 폭발의 최종 방어. **z 크기를 심각도로 읽지
    #   말 것** — 포화값이며, 탐지 여부만 의미가 있다

    # ------------------------------------------------------------------
    def mad_floor(self, feature: str) -> float:
        return float(self.mad_floors.get(feature, self.mad_floor_default))

    def view(self) -> dict:
        """내부 연산 모듈이 쓰는 dict 뷰 (기존 검증된 코어의 인터페이스 유지)."""
        return {
            "sampling": {
                "ribbon_half_w": self.ribbon_half_w,
                "win_frac": self.win_frac,
                "win_min_px": self.win_min_px,
                "win_max_px": self.win_max_px,
                "margin_px": self.margin_px,
                "grad_sigma_px": self.grad_sigma_px,
                "noise_patch_px": self.noise_patch_px,
                "peak_suppress": self.peak_suppress,
                "npk_ratio": self.npk_ratio,
                "delta_sigma_sweep": tuple(self.delta_sigma_sweep),
            },
            "sanitize": {
                "min_cnr_for_ratio": self.min_cnr_for_ratio,
                "min_rise_px": self.min_rise_px,
            },
            "boundary": {"label_inset_px": self.label_inset_px},
            "sequence": {
                "local_window": self.local_window,
                "method": self.method,
                "min_seq_len": self.min_seq_len,
            },
            "features": {"direction": dict(self.direction_overrides)},
        }
