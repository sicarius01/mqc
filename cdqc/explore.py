"""cdqc explore — 플래그된 CD의 피쳐 공간 클러스터링 (실패 모드 발견용).

판정 경로에 관여하지 않는다 (spec §4.4). 선택 의존성: scikit-learn 또는
hdbscan. 없으면 E-EXPL-01.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config
from .errors import CdqcError
from .pipeline import run_pipeline


def run_explore(cfg: Config) -> str:
    res = run_pipeline(cfg)
    l3 = res["l3"]
    flagged = l3.filter(pl.col("flag"))
    if flagged.height < 10:
        return f"플래그 CD {flagged.height}개 — 클러스터링에 부족 (>=10 필요)"

    zcols = [c for c in flagged.columns if c.startswith("z_")]
    X = np.column_stack([flagged[c].to_numpy() for c in zcols])
    X = np.nan_to_num(np.clip(X, -8, 8), nan=0.0)

    labels = None
    algo = ""
    try:
        import hdbscan  # type: ignore
        labels = hdbscan.HDBSCAN(min_cluster_size=max(5, flagged.height // 50)) \
                        .fit_predict(X)
        algo = "hdbscan"
    except ImportError:
        try:
            from sklearn.cluster import KMeans
            k = int(np.clip(flagged.height // 30 + 2, 2, 8))
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=cfg.seed()).fit_predict(X)
            algo = f"kmeans(k={k})"
        except ImportError:
            raise CdqcError("E-EXPL-01") from None

    out = flagged.with_columns(pl.Series("cluster", labels))
    idir = cfg.path("output_dir") / cfg["report"]["internal_dir"]
    idir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(idir / "explore_clusters.parquet")

    lines = [f"# cdqc explore — {algo}, 플래그 CD {flagged.height}개"]
    for c in sorted(set(labels)):
        part = out.filter(pl.col("cluster") == c)
        # 클러스터 특징: 평균 directed z 상위 피쳐
        means = {col[2:]: float(np.nanmean(part[col].to_numpy())) for col in zcols}
        top = sorted(means.items(), key=lambda kv: -kv[1])[:3]
        reasons = part.group_by("reason_code").len().sort("len", descending=True)
        rtxt = ", ".join(f"{r}={n}" for r, n in reasons.rows())
        ttxt = ", ".join(f"{k}:{v:+.1f}" for k, v in top)
        lines.append(f"cluster {c:>3}: n={part.height:<5} top_z=[{ttxt}]  사유=[{rtxt}]")
    return "\n".join(lines)
