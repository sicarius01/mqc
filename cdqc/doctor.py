"""cdqc doctor — 환경·스키마·좌표 점검. 사내 실행 첫날 필수 (spec §6.1).

핵심은 좌표 컨벤션 자동 탐지: 후보 8개 {xy/rowcol} × {y_flip} × {origin 0/1}
각각에 대해 보고 좌표 위치의 평균 그래디언트 크기를 재고, 1등/2등 비 ≥ 3이면
채택해 cache_dir/convention.json에 저장한다. 확인 전까지 어떤 이미지 피쳐도
신뢰하지 않는다 (spec §2.3).
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import map_coordinates

from . import io
from .config import Config
from .errors import ERROR_CODES, CdqcError, warn
from .features.l1 import _grad_mag

ADOPT_RATIO = 3.0
WARN_RATIO = 1.5
MAX_SAMPLE_IMAGES = 30

_PACKAGES = ("numpy", "scipy", "polars", "opencv-python-headless",
             "matplotlib", "pandas", "pyarrow")


def _candidates(scale: float) -> list[dict]:
    out = []
    for convention in ("xy", "rowcol"):
        for y_flip in (False, True):
            for origin in ("zero", "one"):
                out.append({"convention": convention, "origin": origin,
                            "y_flip": y_flip, "scale": scale})
    return out


def _cand_name(c: dict) -> str:
    return f"{c['convention']}/origin={c['origin']}/y_flip={int(c['y_flip'])}"


def detect_convention(cfg: Config, records: pl.DataFrame) -> tuple[dict | None, str]:
    """후보 점수표 계산. (채택된 컨벤션 or None, 리포트 텍스트).

    점수 = 보고 좌표에서 세그먼트 축 방향으로 ±3px 안의 그래디언트 피크까지
    거리(|delta|, px)의 중앙값. 맞는 컨벤션이면 좌표가 엣지 위에 있으므로 ~0,
    origin이 1px 어긋나면 ~1, 축이 뒤집히면 크거나 OOB — "평균 그래디언트
    크기"보다 훨씬 날카롭게 갈린다 (엣지 rise가 1px보다 넓으면 크기 차이는
    원리적으로 작다).
    """
    scale = float(cfg["data"]["coords"]["scale"])
    cands = _candidates(scale)
    offsets = np.arange(-3.0, 3.0 + 0.25, 0.25)
    oob_penalty = 3.0
    deltas: list[list[float]] = [[] for _ in cands]

    all_ids = records["image_id"].unique().sort().to_list()
    step = max(1, len(all_ids) // MAX_SAMPLE_IMAGES)
    image_ids = all_ids[::step][:MAX_SAMPLE_IMAGES]   # 전체에 고르게 분산 표본
    sub = records.filter(pl.col("image_id").is_in(image_ids))
    for (_, image_id), part in sub.partition_by(["recipe_id", "image_id"],
                                                as_dict=True).items():
        img = io.load_image(io.resolve_image_path(cfg, part[0, "image_path"]))
        grad = _grad_mag(img.astype(np.float64))
        h, w = img.shape
        sx_r, sy_r = part["sx"].to_numpy(), part["sy"].to_numpy()
        ex_r, ey_r = part["ex"].to_numpy(), part["ey"].to_numpy()
        for i, c in enumerate(cands):
            sx, sy = io.transform_coords(sx_r, sy_r, c, img.shape)
            ex, ey = io.transform_coords(ex_r, ey_r, c, img.shape)
            seg = np.stack([ex - sx, ey - sy], axis=1)
            norm = np.linalg.norm(seg, axis=1, keepdims=True)
            u = seg / np.where(norm > 0, norm, 1.0)
            pts = np.concatenate([np.stack([sx, sy], 1), np.stack([ex, ey], 1)])
            us = np.concatenate([u, u])
            lat = np.stack([-us[:, 1], us[:, 0]], axis=1)  # 엣지 접선(수직) 방향
            lat_off = np.arange(-2.0, 2.0 + 1.0, 1.0)
            # (n_pts, n_off, n_lat, 2): 축방향 오프셋 × 접선 리본 평균
            P = (pts[:, None, None, :]
                 + us[:, None, None, :] * offsets[None, :, None, None]
                 + lat[:, None, None, :] * lat_off[None, None, :, None])
            x, y = P[..., 0].ravel(), P[..., 1].ravel()
            inb = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
            g = np.zeros(len(x))
            if inb.any():
                g[inb] = map_coordinates(grad, [y[inb], x[inb]], order=1,
                                         mode="constant")
            G = g.reshape(len(pts), len(offsets), len(lat_off)).mean(axis=2)
            valid = inb.reshape(len(pts), len(offsets), len(lat_off)) \
                       .all(axis=(1, 2))
            best = np.abs(offsets[np.argmax(G, axis=1)])
            best[~valid] = oob_penalty
            deltas[i].extend(best.tolist())

    med = np.array([np.median(d) if d else oob_penalty for d in deltas])
    order = np.argsort(med)
    s1 = max(float(med[order[0]]), 0.15)   # 지터/양자화 바닥
    ratio = float(med[order[1]]) / s1

    lines = ["# 좌표 컨벤션 점수표 (보고 좌표 ↔ 축방향 그래디언트 피크 중앙값 거리, px — 낮을수록 좋음)",
             f"# 표본 이미지 {len(image_ids)}개", ""]
    for i in order:
        mark = " <-- 1등" if i == order[0] else ""
        lines.append(f"{_cand_name(cands[i]):<34} {med[i]:>8.3f}{mark}")
    lines.append("")
    lines.append(f"2등/1등 비 = {ratio:.2f}  (채택 기준 >= {ADOPT_RATIO}, "
                 f"경고 기준 < {WARN_RATIO})")

    adopted = None
    if ratio >= ADOPT_RATIO:
        adopted = cands[order[0]]
        lines.append(f"채택: {_cand_name(adopted)}")
    elif ratio < WARN_RATIO:
        warn("W-CONV-01", f"ratio={ratio:.2f}")
        lines.append(f"[W-CONV-01] {ERROR_CODES['W-CONV-01']}")
    else:
        lines.append("채택 보류 — 점수표를 눈으로 확인하고 [data.coords]를 수동 설정할 것")
    return adopted, "\n".join(lines)


def check_quantization(records: pl.DataFrame) -> str:
    """좌표 소수부 분포 → 양자화 경고 (delta 분해능 하한)."""
    coords = np.concatenate([records[c].to_numpy()
                             for c in ("sx", "sy", "ex", "ey")])
    frac = np.mod(coords, 1.0)
    at_int = float(np.mean(np.minimum(frac, 1 - frac) < 0.01))
    at_half = float(np.mean(np.abs(frac - 0.5) < 0.01))
    lines = ["# 좌표 양자화 검사 (소수부 분포)",
             f"정수 근처 비율: {at_int:.3f}   반픽셀 근처 비율: {at_half:.3f}"]
    if at_int > 0.8:
        warn("W-CONV-02", f"integer-quantized coords ({at_int:.2f})")
        lines.append(f"[W-CONV-02] 정수 양자화 — delta 분해능 하한 ≈ 0.5 px")
    elif at_int + at_half > 0.8:
        warn("W-CONV-02", f"half-pixel quantized coords")
        lines.append(f"[W-CONV-02] 반픽셀 양자화 — delta 분해능 하한 ≈ 0.25 px")
    else:
        lines.append("양자화 징후 없음 (subpixel 좌표)")
    return "\n".join(lines)


def sequence_report(records: pl.DataFrame) -> str:
    """카테고리별 n_cd 분포와 cd_index 연속성."""
    lines = ["# 시퀀스 점검", "",
             f"{'recipe':<12}{'category':<10}{'n_seq':>6}{'n_cd min/med/max':>20}{'index gaps':>12}"]
    for (recipe, cat), part in records.partition_by(
            ["recipe_id", "category_id"], as_dict=True).items():
        ns, gaps = [], 0
        for _, seq in part.partition_by(["image_id"], as_dict=True).items():
            idx = np.sort(seq["cd_index"].to_numpy())
            ns.append(len(idx))
            gaps += int(np.sum(np.diff(idx) > 1))
        ns = np.array(ns)
        lines.append(f"{recipe:<12}{cat:<10}{len(ns):>6}"
                     f"{f'{ns.min()}/{int(np.median(ns))}/{ns.max()}':>20}{gaps:>12}")
    return "\n".join(lines)


def env_report() -> str:
    lines = ["# 환경", f"python {sys.version.split()[0]}"]
    for p in _PACKAGES:
        try:
            lines.append(f"{p} {importlib.metadata.version(p)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{p} MISSING")
    lines.append("런타임 네트워크 호출: 없음 (설계상 0 — cdqc는 소켓을 열지 않는다)")
    return "\n".join(lines)


def run_doctor(cfg: Config) -> tuple[str, bool]:
    """전체 점검 실행. (리포트 텍스트, 치명 문제 없음 여부)."""
    sections = [env_report()]
    ok = True

    try:
        records = io.load_records(cfg)
        sections.append(f"# 스키마\n레코드 {records.height}건, "
                        f"이미지 {records['image_id'].n_unique()}개, "
                        f"레시피 {records['recipe_id'].n_unique()}개 — OK")
    except CdqcError as e:
        sections.append(f"# 스키마\nFAIL {e}")
        report = "\n\n".join(sections)
        return report, False

    # 이미지 표본 점검은 load_image가 detect 단계에서 같이 수행 (E-DATA-04/05)
    try:
        adopted, conv_text = detect_convention(cfg, records)
        sections.append(conv_text)
        summary_dir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "convention.txt").write_text(conv_text, encoding="utf-8")
        if adopted is not None:
            cache = cfg.path("cache_dir")
            cache.mkdir(parents=True, exist_ok=True)
            (cache / io.CONVENTION_CACHE).write_text(
                json.dumps({k: adopted[k] for k in ("convention", "origin", "y_flip")}),
                encoding="utf-8")
        elif cfg["data"]["coords"]["convention"] == "auto":
            ok = False
    except CdqcError as e:
        sections.append(f"# 좌표 컨벤션\nFAIL {e}")
        ok = False

    sections.append(check_quantization(records))
    sections.append(sequence_report(records))
    report = "\n\n".join(sections)

    summary_dir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "doctor.txt").write_text(report, encoding="utf-8")
    return report, ok
