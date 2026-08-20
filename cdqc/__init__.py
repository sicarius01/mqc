"""cdqc — TEM CD 측정 품질 판정용 연산 라이브러리.

순수 함수 모음: 정해진 포맷의 데이터(정렬된 좌표 배열, uint8 이미지, nm 단위)
→ 피쳐/robust 통계/z 연산 → 리턴. 파일 I/O·워크플로우·판정 기준은 전부
사용자(사내) 몫이다. 사용법은 README.md, 설계는 cdqc_spec.md.

    import cdqc
    p = cdqc.Params()
    l3 = cdqc.extract_l3(img, S, E, px_nm=0.5, value_nm=v, params=p)
    stats = cdqc.cohort_stats(l3_normal, params=p)
    z = cdqc.apply_z(l3, stats, params=p)
    top = cdqc.top_feature(z)
"""

from .api import (apply_z, cohort_stats, extract_l1, extract_l2, extract_l3,
                  hist_emd, impact_nm, max_run, robust_stats,
                  threshold_from_quantile, top_feature)
from .errors import ERROR_CODES, CdqcError
from .features.registry import BY_NAME, REASONS, REGISTRY, Z_ON_BAD
from .params import Params
from .utils import (convention_candidates, convention_scores, infer_px_nm,
                    normalize_unit, ratio_cv, to_nm, transform_coords)

FEATURES = REGISTRY   # 피쳐 메타데이터 공개 별칭 (이름/worse_when/사유코드/설명)

__version__ = "0.2.0"

__all__ = [
    "Params",
    "extract_l3", "extract_l2", "extract_l1",
    "cohort_stats", "apply_z", "robust_stats", "threshold_from_quantile",
    "top_feature", "impact_nm", "max_run", "hist_emd",
    "transform_coords", "convention_scores", "convention_candidates",
    "normalize_unit", "to_nm", "infer_px_nm", "ratio_cv",
    "FEATURES", "REGISTRY", "BY_NAME", "REASONS", "Z_ON_BAD",
    "CdqcError", "ERROR_CODES",
]
