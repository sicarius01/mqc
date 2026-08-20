"""연산 파라미터 — config.toml을 대체하는 단일 dataclass.

모든 공개 API 함수가 Params 하나를 받는다. 기본값은 여기 내장돼 있고,
사용자는 필드를 바꾼 인스턴스를 만들어 쓰면 된다. 저장/로드가 필요하면
dataclasses.asdict()로 사용자가 알아서 한다 — cdqc는 파일을 만지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_mad_floors() -> dict[str, float]:
    # 이산/퇴화 피쳐의 MAD=0 방어 (spec §4.2). 단위: 피쳐 각자의 단위
    return {
        "npk_s": 0.5, "npk_e": 0.5,
        "n_cd": 0.5,
        "sat_lo": 0.002, "sat_hi": 0.002,          # 픽셀 비율
        "hist_emd": 0.5,
        "frac_flagged": 0.05,
        "overshoot_s": 0.02, "overshoot_e": 0.02,
        "max_run": 0.5,
        "noise_sigma": 0.05,                        # gray level (uint8 양자화 방어)
        "dyn_range": 1.0,
        "delta_median_s": 0.05, "delta_median_e": 0.05,   # nm (subpixel 분해능)
        "value_mismatch_nm": 0.02,                  # nm
        "mdist_s": 0.05, "mdist_e": 0.05,           # nm (마스크 경계 픽셀 양자화)
        "mask_n_components": 0.5,                   # 정수
        "mask_hole_frac": 0.002,
        "mask_area_frac": 0.002,
        "angle_spread": 0.05,                       # deg
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

    # ---- 시퀀스 잔차 ------------------------------------------------------
    local_window: int = 9         # 국소 추세 창 (홀수). 구조 변화 스케일보다 짧게
    method: str = "robust_linear"  # robust_linear(국소 Theil-Sen) | hampel(러닝 메디안)
    min_seq_len: int = 5          # 미만이면 시퀀스 잔차 NaN

    # ---- 코호트 통계 / z --------------------------------------------------
    trim_frac: float = 0.05       # 오염 방어 트림 (1차 z 상위 이 비율 제거 후 재계산)
    mad_floor_default: float = 1e-6
    mad_floors: dict[str, float] = field(default_factory=_default_mad_floors)
    log_features: tuple[str, ...] = ("rise_s", "rise_e", "margin_s", "margin_e",
                                     "dstep_s", "dstep_e", "cd_mad")
    direction_overrides: dict[str, str] = field(default_factory=dict)
    # ^ 피쳐별 worse_when 오버라이드 ("low"|"high"|"both"). 기본은 registry

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
            },
            "sequence": {
                "local_window": self.local_window,
                "method": self.method,
                "min_seq_len": self.min_seq_len,
            },
            "features": {"direction": dict(self.direction_overrides)},
        }
