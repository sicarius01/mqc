"""cdqc evaluate — labels.csv 기반 탐지/국소화/ablation 평가 (spec §6.7).

labels.csv 스키마: image_id, label(0/1), bad_cd_indices(선택, ';' 구분).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config
from .errors import CdqcError
from .features.registry import BY_NAME
from .io import read_nasca_csv
from .pipeline import run_pipeline

_GEOM_REASONS = {"SEQUENCE_JUMP", "GEOMETRY_ODD"}


def _load_labels(cfg: Config) -> pl.DataFrame:
    path = cfg.path("labels")
    if not path.exists():
        raise CdqcError("E-EVAL-01", str(path))
    df = pl.from_pandas(read_nasca_csv(path))
    if "image_id" not in df.columns or "label" not in df.columns:
        raise CdqcError("E-EVAL-02", f"columns={df.columns}")
    df = df.with_columns(pl.col("image_id").cast(pl.Utf8),
                         pl.col("label").cast(pl.Int64))
    if "bad_cd_indices" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("bad_cd_indices"))
    return df


def _image_scores(res: dict, feature_subset: set[str] | None) -> pl.DataFrame:
    """이미지별 연속 점수. subset이 주어지면 해당 L3 피쳐 z만 사용 (ablation)."""
    l3, l2, l1 = res["l3"], res["l2"], res["l1"]
    thr = res["thresholds"]

    zcols = [c for c in l3.columns if c.startswith("z_")
             and (feature_subset is None or c[2:] in feature_subset)]
    cd_scores = (l3.with_columns(
        pl.max_horizontal([pl.col(c) for c in zcols]).alias("_s"))
        .group_by("image_id").agg((pl.col("_s").max() / thr["t_soft"]).alias("s_cd")))

    seq = l2.group_by("image_id").agg(
        (pl.col("impact_nm") / pl.col("tolerance_nm")).max().alias("s_impact"))
    g2cols = [f"z_{f.name}" for f in BY_NAME.values()
              if f.level == "l2" and f.g2 and f"z_{f.name}" in l2.columns]
    seq2 = l2.with_columns(
        pl.max_horizontal([pl.col(c) for c in g2cols]).alias("_g2")) \
        .group_by("image_id").agg((pl.col("_g2").max() / thr["t_seq"]).alias("s_seq"))
    g0cols = [f"z_{f.name}" for f in BY_NAME.values()
              if f.level == "l1" and f.g0 and f"z_{f.name}" in l1.columns]
    img = l1.with_columns(
        pl.max_horizontal([pl.col(c) for c in g0cols]).alias("_g0")) \
        .select("image_id", (pl.col("_g0") / thr["t_image"]).alias("s_img"))

    return (cd_scores.join(seq, on="image_id", how="full", coalesce=True)
            .join(seq2, on="image_id", how="full", coalesce=True)
            .join(img, on="image_id", how="full", coalesce=True))


def _recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float(np.mean(pos > thr))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def run_evaluate(cfg: Config) -> str:
    labels = _load_labels(cfg)
    res = run_pipeline(cfg)
    fpr = float(cfg.get("evaluate.fpr", 0.05))

    geom = {n for n, f in BY_NAME.items() if f.level == "l3" and f.reason in _GEOM_REASONS}
    variants = {
        "L3-geom만": (geom, False),
        "+image evidence": (None, False),
        "+L1/L2 게이트": (None, True),
    }
    lines = ["# cdqc evaluate", f"라벨 {labels.height}건 "
             f"(bad {int((labels['label'] == 1).sum())}), 고정 FPR {fpr}"]

    lab = labels.select("image_id", "label")
    for name, (subset, full) in variants.items():
        sc = _image_scores(res, subset)
        expr = [c for c in ("s_cd", "s_impact", "s_seq", "s_img")
                if (full or c in ("s_cd",)) and c in sc.columns]
        sc = sc.with_columns(pl.max_horizontal([pl.col(c) for c in expr])
                             .fill_null(-np.inf).alias("score"))
        j = lab.join(sc.select("image_id", "score"), on="image_id", how="left") \
               .with_columns(pl.col("score").fill_null(-np.inf))
        s = j["score"].to_numpy()
        y = j["label"].to_numpy()
        lines.append(f"{name:<18} recall@fpr={_recall_at_fpr(s, y, fpr):.3f}  "
                     f"AUC={_auc(s, y):.3f}")

    # 국소화: 불량으로 맞힌 이미지 중 최고 z CD가 지목 CD와 일치하는 비율
    with_idx = labels.filter((pl.col("label") == 1)
                             & pl.col("bad_cd_indices").is_not_null())
    if with_idx.height:
        l3 = res["l3"]
        hits = total = 0
        for row in with_idx.iter_rows(named=True):
            bad = {int(v) for v in str(row["bad_cd_indices"]).split(";") if v.strip()}
            part = l3.filter(pl.col("image_id") == row["image_id"])
            if part.height == 0 or not bad:
                continue
            top = part.sort("top_z", descending=True, nulls_last=True).row(0, named=True)
            total += 1
            hits += int(top["cd_index"] in bad)
        if total:
            lines.append(f"국소화 일치율: {hits}/{total} = {hits / total:.3f}")

    out = "\n".join(lines)
    sdir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "evaluate.txt").write_text(out, encoding="utf-8")
    return out
