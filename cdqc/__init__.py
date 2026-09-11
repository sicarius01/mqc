"""cdqc — TEM CD 측정 품질 판정용 연산 라이브러리.

순수 함수 모음: 정해진 포맷의 데이터(정렬된 좌표 배열, uint8 이미지, nm 단위)
→ 피쳐/robust 통계/z 연산 → 리턴. 파일 I/O·워크플로우·판정 기준은 전부
사용자(사내) 몫이다. 사용법은 README.md, 설계는 cdqc_spec.md.

    import cdqc
    p = cdqc.Params()
    l3 = cdqc.extract_l3(img, S, E, px_nm=0.5, value_nm=v, params=p)
    l3 = l3.join(cdqc.boundary_features(cdqc.mask_maps(labelmap), S, E, 0.5))
    z = cdqc.cohort_z(l3_all, ["category_id"], base_mask=is_normal, params=p)
    agg = cdqc.aggregate_z(z)            # z_max / z_top2 / z_top3 / n_z_valid
    top = cdqc.top_feature(z)
"""

from .api import (ablation_table, apply_z, boundary_features, cohort_stats,
                  extract_l1, extract_l2, extract_l3, extract_mask_image,
                  extract_mask_l3, flag_rollup, hist_emd, impact_nm,
                  localization_hit, localization_rate, mask_maps, max_run,
                  recall_at_fpr, robust_stats, threshold_from_quantile,
                  top_feature)
from .errors import ERROR_CODES, CdqcError
from .features.registry import BY_NAME, REASONS, REGISTRY, Z_ON_BAD
from .normalize import (abs_flags, agg_columns, aggregate_z, cohort_z,
                        floor_stats, mode_flags)
from .params import Params
from .utils import (close_annotation, convention_candidates, convention_scores,
                    infer_px_nm, normalize_unit, ratio_cv, to_nm, to_uint8,
                    transform_coords)
from .validate import (INJECTIONS, category_summary, inject_coords,
                       injection_test, select_grad_sigma,
                       trajectory_check)

FEATURES = REGISTRY   # 피쳐 메타데이터 공개 별칭 (이름/worse_when/사유코드/설명)

__version__ = "0.6.0"

__all__ = [
    "Params",
    "extract_l3", "extract_l2", "extract_l1",
    "mask_maps", "boundary_features",
    "extract_mask_l3", "extract_mask_image",
    "cohort_stats", "apply_z", "robust_stats", "threshold_from_quantile",
    "cohort_z", "aggregate_z", "agg_columns", "floor_stats",
    "abs_flags", "mode_flags",
    "top_feature", "impact_nm", "max_run", "flag_rollup", "hist_emd",
    "recall_at_fpr", "localization_hit", "localization_rate", "ablation_table",
    "inject_coords", "injection_test", "select_grad_sigma",
    "trajectory_check", "category_summary",
    "INJECTIONS",
    "transform_coords", "convention_scores", "convention_candidates",
    "normalize_unit", "to_nm", "to_uint8", "infer_px_nm", "ratio_cv",
    "close_annotation",
    "FEATURES", "REGISTRY", "BY_NAME", "REASONS", "Z_ON_BAD",
    "CdqcError", "ERROR_CODES",
]
