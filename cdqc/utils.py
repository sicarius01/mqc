"""좌표 컨벤션·단위·px_nm 유틸 — 전부 순수 함수, 판단은 사용자 몫.

내부 좌표 컨벤션: (x, y) = (col, row), zero-origin, 이미지 픽셀 단위 float.
extract_* 함수들은 이 컨벤션의 좌표만 받는다. 원본 좌표가 다른 컨벤션이면
transform_coords()로 변환해서 넣는다 (후보 비교는 convention_scores()).
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .errors import CdqcError

_ANGSTROM_TOKENS = {"å", "a", "angstrom", "ang"}
_NM_TOKENS = {"nm", "nanometer", "nanometre"}


# ---------------------------------------------------------------- 단위

def normalize_unit(s) -> str | None:
    """단위 문자열 → "angstrom" | "nm" | None(인식 불가).

    NFKC 정규화로 U+00C5(Å)/U+212B(ANGSTROM SIGN)/NFD("A"+U+030A)를 통일한
    뒤 소문자 비교. 목록 밖 값은 None — 호출부가 추측 없이 중단해야 한다.
    """
    if s is None:
        return None
    t = unicodedata.normalize("NFKC", str(s).strip()).lower()
    if t in _ANGSTROM_TOKENS:
        return "angstrom"
    if t in _NM_TOKENS:
        return "nm"
    return None


def to_nm(value, unit) -> np.ndarray:
    """측정값 → nm. unit은 스칼라 문자열 또는 행별 배열.

    알 수 없는 단위는 발견된 고유값 목록과 함께 E-ARG-03으로 즉시 중단.
    """
    v = np.asarray(value, dtype=np.float64)
    if np.isscalar(unit) or isinstance(unit, str):
        units = np.full(v.shape, unit, dtype=object)
    else:
        units = np.asarray(unit, dtype=object)
    norm = np.array([normalize_unit(u) for u in units.ravel()], dtype=object)
    unknown = sorted({str(u) for u, n in zip(units.ravel(), norm) if n is None})
    if unknown:
        raise CdqcError("E-ARG-03", f"unknown unit: {unknown[0]!r} — 고유값: {unknown}")
    factor = np.where(norm == "angstrom", 0.1, 1.0).reshape(v.shape)
    return v * factor


# ---------------------------------------------------------------- px_nm

def infer_px_nm(S: np.ndarray, E: np.ndarray, value_nm: np.ndarray) -> float:
    """이미지 하나 분량의 (S, E, value_nm)로 px_nm 역산 — value/기하 비의 중앙값.

    주의: 이미지 전체가 균일하게 밀린 계통 편향은 여기 흡수된다 (역산의
    본질적 한계). 행별 비의 산포는 ratio_cv()로 점검할 것.
    """
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    v = np.asarray(value_nm, dtype=np.float64)
    cd_px = np.linalg.norm(E - S, axis=1)
    ok = (cd_px > 1e-9) & np.isfinite(v)
    if not ok.any():
        raise CdqcError("E-ARG-04", "유효한 (좌표, value) 쌍이 없음")
    return float(np.median(v[ok] / cd_px[ok]))


def ratio_cv(S: np.ndarray, E: np.ndarray, value_nm: np.ndarray) -> float:
    """value/기하 비의 robust CV. >0.01이면 좌표와 보고값이 따로 계산됐을 가능성."""
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    v = np.asarray(value_nm, dtype=np.float64)
    cd_px = np.linalg.norm(E - S, axis=1)
    ok = (cd_px > 1e-9) & np.isfinite(v)
    r = v[ok] / cd_px[ok]
    if len(r) < 3:
        return np.nan
    med = np.median(r)
    return float(1.4826 * np.median(np.abs(r - med)) / max(abs(med), 1e-12))


# ---------------------------------------------------------------- 좌표 컨벤션

def transform_coords(xs, ys, conv: dict, shape: tuple[int, int]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """원본 좌표쌍 → 내부 (x=col, y=row), zero-origin, scale 적용.

    conv: {convention: "xy"|"rowcol", origin: "zero"|"one", y_flip: bool, scale: float}
    shape: 이미지 (H, W) — y_flip에 필요.
    """
    a = np.asarray(xs, dtype=np.float64) * float(conv.get("scale", 1.0))
    b = np.asarray(ys, dtype=np.float64) * float(conv.get("scale", 1.0))
    if conv.get("origin", "zero") == "one":
        a = a - 1.0
        b = b - 1.0
    if conv.get("convention", "xy") == "rowcol":
        x, y = b.copy(), a.copy()
    else:
        x, y = a, b
    if conv.get("y_flip", False):
        y = (shape[0] - 1) - y
    return x, y


def convention_candidates(scale: float = 1.0) -> list[dict]:
    """탐지 후보 8개: {xy/rowcol} × {origin 0/1} × {y_flip}."""
    return [{"convention": c, "origin": o, "y_flip": f, "scale": scale}
            for c in ("xy", "rowcol") for f in (False, True)
            for o in ("zero", "one")]


def convention_scores(items: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      scale: float = 1.0) -> pd.DataFrame:
    """좌표 컨벤션 후보 8개의 점수표. **판단(채택)은 사용자 몫.**

    items: 이미지별 (img(uint8 2D), S_raw(n,2), E_raw(n,2)) — 원본 좌표 그대로.
    점수 = 보고 좌표에서 세그먼트 축 방향 ±3px 안의 그래디언트 피크까지
    거리(px, 포물선 subpixel)의 중앙값. 낮을수록 좋음: 맞으면 ~좌표 지터
    수준, origin 1px 오류 ~1, 축 스왑/플립은 큼. 참고 채택 기준:
    1등 ≤ 0.75px AND 2등/1등 비 ≥ 2 (판별력은 좌표 지터에 근본 제한됨).
    """
    import cv2

    cands = convention_candidates(scale)
    offsets = np.arange(-3.0, 3.0 + 0.25, 0.25)
    lat_off = np.arange(-2.0, 3.0, 1.0)
    oob = 3.0
    deltas: list[list[float]] = [[] for _ in cands]

    for img, S_raw, E_raw in items:
        f = img.astype(np.float64)
        gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        grad = np.hypot(gx, gy)
        h, w = img.shape
        S_raw = np.asarray(S_raw, dtype=np.float64)
        E_raw = np.asarray(E_raw, dtype=np.float64)
        for i, c in enumerate(cands):
            sx, sy = transform_coords(S_raw[:, 0], S_raw[:, 1], c, img.shape)
            ex, ey = transform_coords(E_raw[:, 0], E_raw[:, 1], c, img.shape)
            seg = np.stack([ex - sx, ey - sy], axis=1)
            norm = np.linalg.norm(seg, axis=1, keepdims=True)
            u = seg / np.where(norm > 0, norm, 1.0)
            pts = np.concatenate([np.stack([sx, sy], 1), np.stack([ex, ey], 1)])
            us = np.concatenate([u, u])
            lat = np.stack([-us[:, 1], us[:, 0]], axis=1)
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
            valid = inb.reshape(len(pts), len(offsets), len(lat_off)).all(axis=(1, 2))
            imax = np.argmax(G, axis=1)
            step = float(offsets[1] - offsets[0])
            corr = np.zeros(len(pts))
            interior = (imax > 0) & (imax < len(offsets) - 1)
            ii = np.where(interior)[0]
            if len(ii):
                y0, y1, y2 = (G[ii, imax[ii] - 1], G[ii, imax[ii]],
                              G[ii, imax[ii] + 1])
                denom = y0 - 2 * y1 + y2
                ok = np.abs(denom) > 1e-12
                corr[ii[ok]] = np.clip(0.5 * (y0 - y2)[ok] / denom[ok], -0.5, 0.5)
            best = np.abs(offsets[imax] + corr * step)
            best[~valid] = oob
            deltas[i].extend(best.tolist())

    rows = []
    for c, d in zip(cands, deltas):
        rows.append({"convention": c["convention"], "origin": c["origin"],
                     "y_flip": c["y_flip"], "scale": c["scale"],
                     "median_dist_px": float(np.median(d)) if d else oob})
    return (pd.DataFrame(rows).sort_values("median_dist_px")
            .reset_index(drop=True))
