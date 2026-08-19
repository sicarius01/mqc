"""레코드/이미지 로더, 컬럼 매핑, 좌표 변환.

내부 좌표 컨벤션: (x, y) = (col, row), zero-origin, float. 모든 하위 모듈은
이 컨벤션만 본다. 원본 → 내부 변환은 transform_coords() 한 곳에서만 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from .config import Config
from .errors import CdqcError

try:
    # 사용자 소유 파일 (gitignore 대상). 사내에서 func.py.example을 복사해 작성
    from .func import read_nasca_csv
except ModuleNotFoundError:
    from .func_default import read_nasca_csv

STD_COLUMNS = ["recipe_id", "image_id", "image_path", "category_id",
               "cd_index", "sx", "sy", "ex", "ey", "px_nm"]

_STR_COLS = ("recipe_id", "image_id", "image_path", "category_id")
_FLOAT_COLS = ("sx", "sy", "ex", "ey", "px_nm")

# doctor가 좌표 컨벤션 탐지 결과를 기록하는 파일 (cache_dir 기준)
CONVENTION_CACHE = "convention.json"


# ---------------------------------------------------------------- 레코드

def load_records(cfg: Config, path: str | Path | None = None) -> pl.DataFrame:
    """측정 레코드 로드 → 표준 컬럼명, 타입 캐스팅, 스키마 검증.

    path 생략 시 [paths].data_dir에서 format에 맞는 파일 전부를 이어붙인다.
    표준 외 컬럼(합성 truth 등)은 보존한다.
    """
    fmt = cfg["data"]["format"]
    if path is not None:
        files = [Path(path)]
    else:
        data_dir = cfg.path("data_dir")
        ext = {"csv": ".csv", "parquet": ".parquet"}[fmt]
        files = sorted(data_dir.glob(f"*{ext}")) if data_dir.exists() else []
    if not files:
        raise CdqcError("E-DATA-01", f"format={fmt}, dir={cfg.path('data_dir')}")
    for f in files:
        if not f.exists():
            raise CdqcError("E-DATA-01", str(f))

    frames = []
    for f in files:
        if f.suffix == ".parquet":
            frames.append(pl.read_parquet(f))
        else:
            # 사내 CSV는 보안 프로그램 때문에 반드시 사용자 함수를 경유 (cdqc/func.py)
            frames.append(pl.from_pandas(read_nasca_csv(f)))
    df = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]

    # 컬럼 매핑 (사용자 이름 → 표준 이름)
    mapping = cfg["data"]["columns"]
    rename = {}
    for std, user in mapping.items():
        if user not in df.columns:
            raise CdqcError("E-DATA-02", f"'{user}' (→ {std}) not in {df.columns}")
        if user != std:
            rename[user] = std
    if rename:
        df = df.rename(rename)

    try:
        df = df.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in _STR_COLS]
            + [pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS]
            + [pl.col("cd_index").cast(pl.Int64)]
        )
    except pl.exceptions.PolarsError as e:
        raise CdqcError("E-DATA-03", str(e)) from e

    if df.height == 0:
        raise CdqcError("E-DATA-08")

    # cd_index 중복 검사 (image × category 내)
    dup = (df.group_by(["image_id", "category_id", "cd_index"]).len()
             .filter(pl.col("len") > 1))
    if dup.height > 0:
        r = dup.row(0)
        raise CdqcError("E-DATA-06", f"image={r[0]} category={r[1]} cd_index={r[2]}")

    # px_nm 이미지 내 일관성
    incons = (df.group_by("image_id").agg(pl.col("px_nm").n_unique().alias("nu"))
                .filter(pl.col("nu") > 1))
    if incons.height > 0:
        raise CdqcError("E-DATA-07", f"image={incons.row(0)[0]}")

    return df.sort(["recipe_id", "image_id", "category_id", "cd_index"])


# ---------------------------------------------------------------- 이미지

def resolve_image_path(cfg: Config, image_path: str) -> Path:
    """레코드의 image_path를 절대경로로. root 기준 → image_dir 기준 순으로 시도."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    cand = cfg.root / p
    if cand.exists():
        return cand
    return cfg.path("image_dir") / p


def load_image(path: Path) -> np.ndarray:
    """8-bit grayscale 이미지 로드. shape (H, W), dtype uint8."""
    if not path.exists():
        raise CdqcError("E-DATA-04", str(path))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise CdqcError("E-DATA-05", f"{path}: cv2 read failed")
    if img.dtype != np.uint8 or img.ndim != 2:
        raise CdqcError("E-DATA-05", f"{path}: dtype={img.dtype} ndim={img.ndim}")
    return img


# ---------------------------------------------------------------- 좌표 변환

def get_convention(cfg: Config) -> dict:
    """유효 좌표 컨벤션 dict를 반환.

    config [data.coords].convention이 "auto"면 doctor가 저장한 탐지 결과를 쓴다.
    """
    coords = dict(cfg["data"]["coords"])
    if coords["convention"] != "auto":
        return coords
    cache = cfg.path("cache_dir") / CONVENTION_CACHE
    if not cache.exists():
        raise CdqcError("E-CONV-02", f"expected {cache}; run `cdqc doctor` first")
    detected = json.loads(cache.read_text(encoding="utf-8"))
    coords.update(detected)
    return coords


def transform_coords(xs: np.ndarray, ys: np.ndarray, conv: dict,
                     shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """원본 좌표쌍 → 내부 (x=col, y=row), zero-origin, scale 적용.

    conv: {convention: xy|rowcol, origin: zero|one, y_flip: bool, scale: float}
    shape: 이미지 (H, W) — y_flip에 필요.
    """
    a = np.asarray(xs, dtype=np.float64) * float(conv.get("scale", 1.0))
    b = np.asarray(ys, dtype=np.float64) * float(conv.get("scale", 1.0))
    if conv.get("origin", "zero") == "one":
        a = a - 1.0
        b = b - 1.0
    if conv.get("convention", "xy") == "rowcol":
        x, y = b.copy(), a.copy()   # 첫 값이 row였던 것
    else:
        x, y = a, b
    if conv.get("y_flip", False):
        y = (shape[0] - 1) - y
    return x, y


def check_coords_in_bounds(x: np.ndarray, y: np.ndarray,
                           shape: tuple[int, int], context: str = "") -> None:
    h, w = shape
    bad = (x < -1) | (x > w) | (y < -1) | (y > h)
    if bad.any():
        i = int(np.argmax(bad))
        raise CdqcError("E-CONV-01",
                        f"{context} (x={x[i]:.1f}, y={y[i]:.1f}) vs shape={shape}")
