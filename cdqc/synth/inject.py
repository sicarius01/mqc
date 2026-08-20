"""실패 모드 주입 — 독립적으로 강도 조절 가능 (spec §7.2). 개발 전용.

이미지 주입은 ImageMods로 렌더러에 전달하고, 레코드 주입(좌표 이동/삭제)은
여기서 직접 행을 고친다. 각 케이스는 (failure, strength, sev_rank)로 식별되고
sev_rank 오름차순이 곧 강도 오름차순이다 (selftest 단조성 판정의 축).

affected=1 인 CD가 selftest의 z 측정 대상이다. 이미지 전역 실패는 전 CD,
국소 실패는 주입된 CD만 표시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# SynthParams.inject 키 → 실패 이름
FAILURE_KEYS = {
    "defocus": "defocus",
    "low_contrast": "low_contrast",
    "noise_up": "noise_up",
    "edge_jump_nm": "edge_jump",
    "systematic_bias_nm": "systematic_bias",
    "oblique_deg": "oblique",
    "missing_frac": "missing",
    "double_edge_offset_px": "double_edge",
    "plateau_defect": "plateau_defect",
    "saturation": "saturation",
    "partial_damage": "partial_damage",
    "mask_shift_px": "mask_shift",
    "mask_ragged_p": "mask_ragged",
    "rotated_frame_deg": "rotated_frame",
}

ROTATED_FRAME_CD_DROP = 4   # rotated_frame에서 카테고리당 줄이는 CD 수

_TILE_GRID = 4  # features/l1.py의 타일 분할과 일치


@dataclass
class Case:
    failure: str
    strength: float | bool
    sev_rank: int
    n_images: int


@dataclass
class ImageMods:
    rise_mult: float = 1.0
    contrast_mult: float = 1.0
    noise_mult: float = 1.0
    fringe_gain: float = 0.0
    double_edge_offset: float = 0.0
    sat_mult: float = 1.0
    plateau_blobs: list[tuple[float, float, float]] = field(default_factory=list)
    damage_tiles: list[tuple[int, int, int, int]] = field(default_factory=list)
    mask_shift_px: float = 0.0      # 마스크만 이동 (이미지·좌표는 정상)
    mask_ragged_p: float = 0.0      # 마스크 경계 픽셀 플립 확률


def build_cases(sp) -> list[Case]:
    cases = [Case("none", 0, 0, sp.n_images)]
    for key, strengths in sp.inject.items():
        failure = FAILURE_KEYS.get(key)
        if failure is None:
            raise ValueError(f"unknown inject key: {key}")
        for rank, s in enumerate(strengths):
            cases.append(Case(failure, s, rank, sp.n_images_per_case))
    return cases


def _pick_per_category(rows: list[dict], k_per_cat: int,
                       rng: np.random.Generator, min_gap: int = 3) -> list[int]:
    """카테고리별로 내부(끝 3개 제외) CD를 k개 고른 전역 행 인덱스.

    min_gap으로 시퀀스상 간격을 강제한다 — 인접 CD가 같이 점프하면 dstep의
    min-이웃차분이 0이 되어 주입이 자기 자신을 가려버린다.
    """
    by_cat: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_cat.setdefault(r["category_id"], []).append(i)
    picked = []
    for idxs in by_cat.values():
        interior = idxs[3:-3] if len(idxs) > 8 else list(idxs)
        chosen: list[int] = []
        order = rng.permutation(len(interior))
        for j in order:
            cand = interior[j]
            if all(abs(cand - c) >= min_gap for c in chosen):
                chosen.append(cand)
            if len(chosen) >= k_per_cat:
                break
        picked.extend(chosen)
    return picked


def apply_case(case: Case, scene, rows: list[dict], sp,
               rng: np.random.Generator) -> tuple[ImageMods, list[dict]]:
    """케이스 하나를 (이미지 mods, 수정된 레코드 행들)로 변환."""
    mods = ImageMods()
    for r in rows:
        r["affected"] = 0
    f, v = case.failure, case.strength

    if f == "none":
        pass
    elif f == "defocus":
        mult = float(v) if float(v) >= 1 else 1.0     # 0 = 주입 없음(1x)
        mods.rise_mult = mult
        mods.fringe_gain = 0.10 * max(mult - 1, 0) if scene.fringe else 0.0
        for r in rows:
            r["affected"] = 1
    elif f == "low_contrast":
        mods.contrast_mult = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "noise_up":
        mods.noise_mult = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "edge_jump":
        d_px = float(v) / scene.px_nm
        for i in _pick_per_category(rows, 2, rng):
            rows[i]["sx"] += d_px          # S를 +u(→E, +x) 방향 이동
            rows[i]["affected"] = 1
    elif f == "systematic_bias":
        d_px = float(v) / scene.px_nm
        for r in rows:
            r["sx"] += d_px
            r["affected"] = 1
    elif f == "oblique":
        theta = math.radians(float(v))
        n_cd = max(len({r["cd_index"] for r in rows}), 1)
        for i in _pick_per_category(rows, max(n_cd // 3, 1), rng):
            r = rows[i]
            width = r["ex"] - r["sx"]
            new_ey = r["ey"] + math.tan(theta) * width
            if 5 <= new_ey <= scene.H - 5:
                band = scene.bands[r["band"]]
                r["ey"] = new_ey
                r["ex"] = float(band.x_right(np.array([new_ey]), scene.H)[0])
            r["affected"] = 1
    elif f == "missing":
        frac = float(v)
        drop = set()
        by_cat: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_cat.setdefault(r["category_id"], []).append(i)
        for idxs in by_cat.values():
            k = round(frac * len(idxs))
            if k:
                drop.update(rng.choice(idxs, size=k, replace=False).tolist())
        rows = [r for i, r in enumerate(rows) if i not in drop]
    elif f == "double_edge":
        mods.double_edge_offset = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "plateau_defect":
        n_cd = max(len({r["cd_index"] for r in rows}), 1)
        picked = _pick_per_category(rows, max(n_cd // 3, 1), rng)
        if bool(v):
            for i in picked:
                r = rows[i]
                mx, my = (r["sx"] + r["ex"]) / 2, (r["sy"] + r["ey"]) / 2
                amp = 0.4 * scene.contrast * (1 if rng.random() < 0.5 else -1)
                mods.plateau_blobs.append((mx, my, amp))
                r["affected"] = 1
        else:
            for i in picked:                # 앵커: 같은 선택, 블롭 없음
                rows[i]["affected"] = 1
    elif f == "saturation":
        mods.sat_mult = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "mask_shift":
        mods.mask_shift_px = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "mask_ragged":
        mods.mask_ragged_p = float(v)
        for r in rows:
            r["affected"] = 1
    elif f == "rotated_frame":
        # 총체적 실패: 기준 각도 오설정 — 시퀀스 전체 회전 배치 + CD 수 변경.
        # 회전된 세그먼트도 밴드 엣지 위에 놓는다 (개별 delta는 정상이어야 함)
        theta = math.radians(float(v))
        if float(v) > 0:
            by_cat: dict[str, list[int]] = {}
            for i, r in enumerate(rows):
                by_cat.setdefault(r["category_id"], []).append(i)
            drop = set()
            for idxs in by_cat.values():
                drop.update(idxs[-ROTATED_FRAME_CD_DROP:])
            rows = [r for i, r in enumerate(rows) if i not in drop]
            for r in rows:
                width = r["ex"] - r["sx"]
                new_ey = r["ey"] + math.tan(theta) * width
                if 5 <= new_ey <= scene.H - 5:
                    band = scene.bands[r["band"]]
                    r["ey"] = new_ey
                    r["ex"] = float(band.x_right(np.array([new_ey]), scene.H)[0])
        for r in rows:
            r["affected"] = 1
    elif f == "partial_damage":
        frac = float(v)
        n_tiles = round(frac * _TILE_GRID * _TILE_GRID)
        if n_tiles:
            th, tw = scene.H // _TILE_GRID, scene.W // _TILE_GRID
            all_tiles = [(r, c) for r in range(_TILE_GRID) for c in range(_TILE_GRID)]
            chosen = rng.choice(len(all_tiles), size=n_tiles, replace=False)
            for t in chosen:
                r, c = all_tiles[int(t)]
                mods.damage_tiles.append((r * th, (r + 1) * th, c * tw, (c + 1) * tw))
        # affected는 렌더 후 generator가 타일 포함 여부로 표시
    else:
        raise ValueError(f"unknown failure: {f}")

    # 정상 레시피 출력 지터 (주입 후, 모든 좌표에). value는 참값 유지
    for r in rows:
        r["sx"] += rng.normal(0, sp.coord_jitter_px)
        r["sy"] += rng.normal(0, sp.coord_jitter_px)
        r["ex"] += rng.normal(0, sp.coord_jitter_px)
        r["ey"] += rng.normal(0, sp.coord_jitter_px)
    return mods, rows
