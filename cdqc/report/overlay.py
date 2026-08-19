"""internal/overlay — 플래그 CD 오버레이 PNG (사내 전용, 좌표·이미지 포함)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from .. import io
from ..config import Config

_GREEN = (80, 200, 80)
_RED = (60, 60, 230)
_YELLOW = (60, 200, 230)


def draw_overlays(cfg: Config, l3: pl.DataFrame, l1: pl.DataFrame,
                  out_dir: Path) -> int:
    """이미지별 세그먼트 오버레이. 반환: 생성 개수 (config 상한 적용)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    max_images = int(cfg["report"]["overlay_max_images"])
    flagged_only = bool(cfg["report"]["overlay_flagged_only"])

    fail_ids = set(l1.filter(pl.col("image_fail"))["image_id"].to_list())
    flag_ids = set(l3.filter(pl.col("flag"))["image_id"].to_list())
    candidates = sorted(fail_ids | flag_ids) if flagged_only \
        else sorted(l3["image_id"].unique().to_list())
    # 실패 이미지 우선
    candidates = (sorted(fail_ids) + [i for i in candidates if i not in fail_ids])[:max_images]

    paths = dict(zip(l3["image_id"].to_list(),
                     l3["image_path"].to_list())) if "image_path" in l3.columns else {}
    n = 0
    for image_id in candidates:
        part = l3.filter(pl.col("image_id") == image_id)
        if part.height == 0:
            continue
        if image_id in paths:
            img_path = io.resolve_image_path(cfg, paths[image_id])
        else:
            # extract 캐시에는 image_path가 없음 — records에서 못 찾으면 스킵
            continue
        img = cv2.cvtColor(io.load_image(img_path), cv2.COLOR_GRAY2BGR)
        for row in part.iter_rows(named=True):
            p1 = (int(round(row["ix_s"])), int(round(row["iy_s"])))
            p2 = (int(round(row["ix_e"])), int(round(row["iy_e"])))
            color = _RED if row["flag"] else _GREEN
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(img, p1, 2, color, -1, cv2.LINE_AA)
            cv2.circle(img, p2, 2, _YELLOW if row["flag"] else color, -1, cv2.LINE_AA)
            if row["flag"]:
                label = f"{row['top_feature']} z={row['top_z']:.1f}"
                cv2.putText(img, label, (p1[0] + 4, p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, _RED, 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{image_id}.png"), img)
        n += 1
    return n
