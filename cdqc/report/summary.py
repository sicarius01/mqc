"""summary/ — 무차원 요약 출력 (반출 여부는 사용자가 규정 보고 판단).

여기 쓰는 파일에는 nm/px/좌표/이미지가 들어가지 않는다. z, 비율, 카운트,
해시만. impact_nm 같은 물리량은 internal/에만 존재한다.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sps

from ..config import Config
from ..features.registry import BY_NAME


def _z_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("z_")]


def write_feature_stats(frames: dict, path: Path) -> None:
    """피쳐별 정규화(z) 분포 요약 — skew, kurtosis, MAD/median, 결측률."""
    lines = ["# 피쳐별 directed z 분포 (무차원)",
             f"{'level':<5}{'feature':<18}{'n':>7}{'miss%':>7}{'median':>9}"
             f"{'MAD':>8}{'skew':>8}{'kurt':>8}{'|z|>3%':>8}"]
    for level in ("l3", "l2", "l1"):
        df = frames.get(level)
        if df is None:
            continue
        for c in _z_columns(df):
            x = df[c].to_numpy()
            n = len(x)
            finite = x[np.isfinite(x)]
            miss = 100 * (1 - len(finite) / max(n, 1))
            if len(finite) < 3:
                lines.append(f"{level:<5}{c[2:]:<18}{n:>7}{miss:>7.1f}"
                             + " " * 41 + "(표본 부족)")
                continue
            med = np.median(finite)
            mad = np.median(np.abs(finite - med))
            lines.append(
                f"{level:<5}{c[2:]:<18}{n:>7}{miss:>7.1f}{med:>9.2f}{mad:>8.2f}"
                f"{sps.skew(finite):>8.2f}{sps.kurtosis(finite):>8.2f}"
                f"{100 * np.mean(np.abs(finite) > 3):>8.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_correlation(l3: pl.DataFrame, path: Path) -> None:
    """L3 피쳐 간 Spearman 상관. |r|>0.9 쌍 강조 → 중복 피쳐 제거 후보."""
    cols = _z_columns(l3)
    X = np.column_stack([l3[c].to_numpy() for c in cols]) if cols else None
    lines = ["# L3 피쳐 Spearman 상관 (|r|>0.9 쌍)"]
    high = []
    if X is not None and X.shape[0] > 10:
        with np.errstate(invalid="ignore"):
            r, _ = sps.spearmanr(X, nan_policy="omit")
        r = np.atleast_2d(r)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if np.isfinite(r[i, j]) and abs(r[i, j]) > 0.9:
                    high.append((cols[i][2:], cols[j][2:], r[i, j]))
    if high:
        for a, b, v in sorted(high, key=lambda t: -abs(t[2])):
            lines.append(f"{a:<18}{b:<18}r={v:+.3f}  <-- enabled_l3에서 하나 제거 검토")
    else:
        lines.append("|r|>0.9 쌍 없음")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_gate_counts(frames: dict, path: Path) -> None:
    l3, l2, l1 = frames["l3"], frames["l2"], frames["l1"]
    lines = ["# 게이트·사유 코드 분포"]
    n_cd = l3.height
    n_flag = int(l3["flag"].sum())
    lines.append(f"CD 플래그율: {n_flag}/{n_cd} = {100 * n_flag / max(n_cd, 1):.1f}%")
    for (reason,), part in l3.filter(pl.col("flag")).partition_by(
            ["reason_code"], as_dict=True).items():
        lines.append(f"  {reason:<20}{part.height:>7}")
    lines.append("")
    top = (l3.filter(pl.col("flag")).group_by("top_feature").len()
             .sort("len", descending=True).head(10))
    lines.append("top_feature 상위:")
    for name, cnt in top.rows():
        lines.append(f"  {name:<20}{cnt:>7}")
    lines.append("")
    n_img = l1.height
    lines.append(f"이미지 FAIL: {int(l1['image_fail'].sum())}/{n_img} = "
                 f"{100 * l1['image_fail'].sum() / max(n_img, 1):.1f}%")
    lines.append(f"  G0(IMAGE_BAD)        : {int(l1['g0_hit'].sum())}")
    lines.append(f"  G1(STAT_CONTAMINATED): {int(l1['g1_hit'].sum())}")
    lines.append(f"  G2(SYSTEMATIC_SHIFT) : {int(l1['g2_hit'].sum())}")
    lines.append("")
    lines.append("이미지 사유 조합:")
    for (rs,), part in l1.filter(pl.col("image_fail")).partition_by(
            ["image_reasons"], as_dict=True).items():
        lines.append(f"  {rs:<45}{part.height:>6}")
    lines.append("")
    lines.append(f"코호트 폴백 사용 CD 비율: "
                 f"{100 * (l3['cohort_fallback_level'] > 0).mean():.1f}%")
    path.write_text("\n".join(lines), encoding="utf-8")


def git_hash(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=root, timeout=10)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def write_version(cfg: Config, path: Path) -> None:
    from .. import __version__
    lines = [
        f"cdqc {__version__}",
        f"git {git_hash(cfg.root)}",
        f"config_hash {cfg.config_hash}",
        f"calibrated_hash {cfg.calibrated_hash}",
        f"run_at {_dt.datetime.now().isoformat(timespec='seconds')}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(cfg: Config, frames: dict) -> Path:
    """[report.summary] 플래그에 따라 summary/*.txt 일괄 작성."""
    sdir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
    sdir.mkdir(parents=True, exist_ok=True)
    flags = cfg["report"]["summary"]
    if flags["feature_stats"]:
        write_feature_stats(frames, sdir / "feature_stats.txt")
    if flags["correlation"]:
        write_correlation(frames["l3"], sdir / "correlation.txt")
    if flags["gate_counts"]:
        write_gate_counts(frames, sdir / "gate_counts.txt")
    if flags["version"]:
        write_version(cfg, sdir / "version.txt")
    return sdir
