"""레코드/이미지 로더, 컬럼 매핑, 단위/px_nm 처리, 좌표 변환.

내부 좌표 컨벤션: (x, y) = (col, row), zero-origin, float. 모든 하위 모듈은
이 컨벤션만 본다. 원본 → 내부 변환은 transform_coords() 한 곳에서만 한다.

입력 계약 (변경 지시 #01): 필수 컬럼은 image_id, image_path, category_id,
sx, sy, ex, ey, value 뿐이다. px_nm은 value에서 역산하고(제공되면 우선),
cd_index는 파일 행 순서로 생성하며(row_order), recipe_id는 없으면 상수로
채운다. 단위 컬럼이 있으면 행별로 nm 변환(value_nm), 알 수 없는 단위는
추측하지 않고 E-DATA-08로 중단한다.
"""

from __future__ import annotations

import json
import unicodedata
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

STD_REQUIRED = ["image_id", "image_path", "category_id",
                "sx", "sy", "ex", "ey", "value"]
STD_OPTIONAL = ["recipe_id", "cd_index", "px_nm", "unit"]

_STR_COLS = ("recipe_id", "image_id", "image_path", "category_id")
_GROUP = ["recipe_id", "image_id", "category_id"]

# doctor가 좌표 컨벤션 탐지 결과를 기록하는 파일 (cache_dir 기준)
CONVENTION_CACHE = "convention.json"

_ANGSTROM_TOKENS = {"å", "a", "angstrom", "ang"}
_NM_TOKENS = {"nm", "nanometer", "nanometre"}


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


# ---------------------------------------------------------------- 레코드

def _read_files(cfg: Config, path: str | Path | None) -> pl.DataFrame:
    fmt = cfg["data"]["format"]
    if path is not None:
        files = [Path(path)]
    else:
        data_dir = cfg.path("data_dir")
        ext = {"csv": ".csv", "parquet": ".parquet"}[fmt]
        # 파일명 정렬 순 → 파일 내 행 순이 곧 측정 순서 (cd_index_source=row_order)
        files = sorted(data_dir.glob(f"*{ext}")) if data_dir.exists() else []
    if not files:
        raise CdqcError("E-DATA-01", f"format={fmt}, dir={cfg.path('data_dir')}")
    frames = []
    for f in files:
        if not f.exists():
            raise CdqcError("E-DATA-01", str(f))
        if f.suffix == ".parquet":
            frames.append(pl.read_parquet(f))
        else:
            # 사내 CSV는 보안 프로그램 때문에 반드시 사용자 함수를 경유 (cdqc/func.py)
            frames.append(pl.from_pandas(read_nasca_csv(f)))
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def _apply_mapping(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """사용자 컬럼명 → 표준 이름. 필수 누락은 E-DATA-02, 선택은 있으면 rename."""
    mapping = cfg["data"]["columns"]
    rename: dict[str, str] = {}
    for std in STD_REQUIRED:
        user = mapping.get(std, std)
        if user not in df.columns:
            raise CdqcError("E-DATA-02", f"'{user}' (→ {std}) not in {df.columns}")
        if user != std:
            rename[user] = std
    for std in STD_OPTIONAL:
        user = mapping.get(std, std)
        if not user:                       # "" = 이 컬럼 없음
            continue
        if user in df.columns and user != std:
            rename[user] = std
    return df.rename(rename)


def _value_nm(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """단위 처리 → value_nm 컬럼. 이후 로직은 value_nm만 본다."""
    unit_mapped = bool(cfg["data"]["columns"].get("unit"))
    if unit_mapped:
        user = cfg["data"]["columns"]["unit"]
        if "unit" not in df.columns:
            raise CdqcError("E-DATA-02", f"'{user}' (→ unit) not in {df.columns}")
        units = df["unit"].cast(pl.Utf8).to_list()
        norm = [normalize_unit(u) for u in units]
        unknown = sorted({str(u) for u, n in zip(units, norm) if n is None})
        if unknown:
            raise CdqcError("E-DATA-08",
                            f"unknown unit: {unknown[0]!r} — 발견된 고유 단위값: {unknown}")
        factor = np.where(np.array(norm) == "angstrom", 0.1, 1.0)
        return df.with_columns(
            (pl.col("value") * pl.Series(factor)).alias("value_nm"))
    fallback = normalize_unit(cfg["data"]["value_unit"])
    if fallback is None:
        raise CdqcError("E-CONF-03",
                        f"data.value_unit={cfg['data']['value_unit']!r}")
    f = 0.1 if fallback == "angstrom" else 1.0
    return df.with_columns((pl.col("value") * f).alias("value_nm"))


def _cd_index(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """cd_index 생성/검증. row_order면 원본 행 순서(__order__) 기준 cumcount."""
    source = cfg["data"]["cd_index_source"]
    if source == "column":
        if "cd_index" not in df.columns:
            raise CdqcError("E-DATA-02",
                            f"cd_index_source=column인데 cd_index 컬럼 없음")
        try:
            df = df.with_columns(pl.col("cd_index").cast(pl.Int64))
        except pl.exceptions.PolarsError as e:
            raise CdqcError("E-DATA-03", str(e)) from e
        dup = (df.group_by(["image_id", "category_id", "cd_index"]).len()
                 .filter(pl.col("len") > 1))
        if dup.height > 0:
            r = dup.row(0)
            raise CdqcError("E-DATA-06",
                            f"image={r[0]} category={r[1]} cd_index={r[2]}")
        return df
    if source != "row_order":
        raise CdqcError("E-CONF-03", f"data.cd_index_source={source!r}")
    # 파일 순서 기반 — 매핑된 cd_index 컬럼이 있어도 무시(덮어씀)
    return df.with_columns(
        (pl.col("__order__").rank("ordinal").over(_GROUP) - 1)
        .cast(pl.Int64).alias("cd_index"))


def _px_nm(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """px_nm: 컬럼이 있으면 우선(하위호환), 없으면 이미지별 value/기하 비 중앙값."""
    if "px_nm" in df.columns:
        df = df.with_columns(pl.col("px_nm").cast(pl.Float64))
        incons = (df.group_by("image_id")
                    .agg(pl.col("px_nm").n_unique().alias("nu"))
                    .filter(pl.col("nu") > 1))
        if incons.height > 0:
            raise CdqcError("E-DATA-07", f"image={incons.row(0)[0]}")
        return df
    scale = float(cfg["data"]["coords"]["scale"])
    cd_px = ((pl.col("ex") - pl.col("sx")).pow(2)
             + (pl.col("ey") - pl.col("sy")).pow(2)).sqrt() * scale
    ratio = (pl.col("value_nm") / cd_px)
    df = df.with_columns(
        pl.when(cd_px > 1e-9).then(ratio).otherwise(None).alias("__ratio__"))
    df = df.with_columns(
        pl.col("__ratio__").median().over("image_id").alias("px_nm"))
    bad = df.filter(pl.col("px_nm").is_null())
    if bad.height > 0:
        raise CdqcError("E-DATA-10", f"image={bad.row(0, named=True)['image_id']}")
    return df.drop("__ratio__")


def load_records(cfg: Config, path: str | Path | None = None) -> pl.DataFrame:
    """측정 레코드 로드 → 표준 컬럼 + value_nm + px_nm + cd_index.

    path 생략 시 [paths].data_dir의 format 파일 전부를 파일명 정렬 순으로
    이어붙인다. 표준 외 컬럼(합성 truth 등)은 보존한다.
    """
    df = _read_files(cfg, path)
    # 원본 행 순서 보존 — cd_index(row_order)의 기준. 정렬/파티션 전에 부여
    df = df.with_row_index("__order__")
    df = _apply_mapping(df, cfg)

    if "recipe_id" not in df.columns:
        df = df.with_columns(
            pl.lit(str(cfg["data"]["default_recipe_id"])).alias("recipe_id"))

    try:
        df = df.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in _STR_COLS if c in df.columns]
            + [pl.col(c).cast(pl.Float64)
               for c in ("sx", "sy", "ex", "ey", "value")])
    except pl.exceptions.PolarsError as e:
        raise CdqcError("E-DATA-03", str(e)) from e

    if df.height == 0:
        raise CdqcError("E-DATA-09")

    df = _value_nm(df, cfg)
    df = _cd_index(df, cfg)
    df = _px_nm(df, cfg)
    return (df.sort(["recipe_id", "image_id", "category_id", "cd_index"])
              .drop("__order__"))


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
