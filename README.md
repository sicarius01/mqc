# cdqc — TEM CD 측정 품질 판정용 연산 라이브러리

DL 세그멘테이션 레시피가 뽑은 CD 측정값 `(sx, sy, ex, ey)`의 품질 판정에
필요한 **연산만** 제공하는 순수 함수 라이브러리다. 피쳐 추출(CD/시퀀스/이미지),
robust 통계, directed z, 판정 헬퍼까지 — 그 이상은 하지 않는다.

- **파일을 읽지 않는다.** CSV/이미지 로딩, 컬럼 정리는 전부 사용자 코드.
- **워크플로우가 없다.** CLI·config 파일·출력 규약 없음. 함수 호출 → 리턴.
- **기준을 정하지 않는다.** 코호트 분리·임계값·공차·판정 정책은 사용자 몫.

설계 문서: `cdqc_spec.md` (피쳐 의미·정규화 원리), `cdqc_change_02_library_api.md`.

## 설치 (사내)

```
git clone https://github.com/sicarius01/mqc.git
cd mqc
pip install -e .
```

- `-e .` = **editable 설치**: 패키지를 복사하지 않고 이 레포 폴더를 직접 참조하게
  등록한다. 이후 어느 경로에서든 `import cdqc`가 되고, **`git pull`만 받으면
  재설치 없이 새 코드가 바로 반영**된다.
- venv를 쓰는 경우: `python -m venv .venv` → `.venv\Scripts\activate` → 위와 동일.
- 의존성(numpy, scipy, pandas, opencv-python-headless — 4개, 버전 핀)은
  `pip install -e .` 가 pyproject.toml을 보고 같이 설치한다.
- 확인: `python -c "import cdqc; print(cdqc.__version__)"` → `0.2.0`

## 입력 계약

| 항목 | 형식 |
|---|---|
| 좌표 `S`, `E` | `(n,2)` float ndarray, **(x=col, y=row), zero-origin, 이미지 픽셀** |
| 시퀀스 | 한 번의 `extract_l3` 호출 = 한 (이미지 × CD 카테고리). **측정 순서대로 정렬**해서 넣을 것 (시퀀스 피쳐가 순서에 의존) |
| 이미지 | `np.ndarray[uint8]`, shape (H, W). `None`이면 기하 피쳐만 |
| 길이 단위 | 함수 경계에서 항상 **nm** (`to_nm`, `infer_px_nm` 유틸 참고) |
| DataFrame | pandas 입출력 |

## 사용 예제 (사내 코드 골격)

```python
import cv2
import numpy as np
import pandas as pd
import cdqc

p = cdqc.Params()                       # 연산 파라미터 (기본값 내장, 필드 수정 가능)

# ── 1) 데이터 준비: 전부 사용자 코드 ─────────────────────────────────
df = my_read_csv(...)                    # 사내 보안 CSV → DataFrame (컬럼 정리 포함)
images = {iid: cv2.imread(pth, cv2.IMREAD_GRAYSCALE) for iid, pth in ...}

# 단위 → nm (Å/nm 자동 인식, 모르는 단위는 즉시 에러)
df["value_nm"] = cdqc.to_nm(df["value"].to_numpy(), df["unit"].to_numpy())

# 좌표 컨벤션이 불확실하면 후보 점수표 (판단은 사람이)
# table = cdqc.convention_scores([(images[i], S_raw, E_raw), ...])

# ── 2) 피쳐 추출: 시퀀스(이미지×카테고리) 단위 호출 ──────────────────
parts = []
for (iid, cat), g in df.groupby(["image_id", "category_id"], sort=False):
    S = g[["sx", "sy"]].to_numpy()       # 측정 순서대로 정렬돼 있어야 함
    E = g[["ex", "ey"]].to_numpy()
    px = cdqc.infer_px_nm(S, E, g["value_nm"].to_numpy())   # 또는 아는 값
    f = cdqc.extract_l3(images[iid], S, E, px,
                        value_nm=g["value_nm"].to_numpy(), params=p)
    parts.append(f.assign(image_id=iid, category_id=cat))
l3 = pd.concat(parts, ignore_index=True)

l2 = cdqc.extract_l2(l3, ["image_id", "category_id"], p)
l1 = pd.DataFrame([{"image_id": iid, **cdqc.extract_l1(img, p)}
                   for iid, img in images.items()])

# ── 3) 정규화: 코호트는 사용자가 자름 (예: 카테고리별) ────────────────
z_parts = []
for cat, g in l3.groupby("category_id"):
    normal = g[...]                      # 정상으로 간주할 행 선택 — 사용자 판단
    stats = cdqc.cohort_stats(normal, params=p)   # 평범한 dict — 저장/로드 자유
    z_parts.append(cdqc.apply_z(g, stats, p))     # z_*(directed) + zs_*(부호) 추가
z = pd.concat(z_parts)

# ── 4) 판정: 기준은 사용자, 연산은 헬퍼 ──────────────────────────────
top = cdqc.top_feature(z)                          # 행별 최대 z + 피쳐 + 사유코드
t_soft = cdqc.threshold_from_quantile(top_normal["top_z"], 0.90)
flag = top["top_z"] > t_soft
impact = cdqc.impact_nm(seq["cd_nm"], seq_flags)   # 공차(nm)와 직접 비교
run = cdqc.max_run(seq_flags)                      # 뭉침 vs 산발
```

### 마스크(DL segmentation) 흐름 — 마스크가 있는 카테고리만

마스크 PNG를 읽고 클래스별 이진 분리하는 것까지가 사용자 몫이다
(해상도가 이미지와 다르면 E-ARG-07 — 리사이즈도 명시적으로 사용자가).

```python
mask_rgb = cv2.imread(mask_path)                    # 사용자: 파일 읽기
mask = (mask_rgb[..., 0] == CLASS_COLOR_R)          # 사용자: 클래스 → 이진 분리

for (iid, cat), g in df.groupby(["image_id", "category_id"], sort=False):
    f = cdqc.extract_l3(images[iid], S, E, px, params=p)
    fm = cdqc.extract_mask_l3(masks[(iid, cat)], images[iid], S, E, px, params=p)
    f = pd.concat([f, fm], axis=1)      # 행 순서가 같아 index join — mdist/mgrad/minside
    ...

# 마스크 자체의 형태 이상 (이미지×클래스당 한 행 — L1과 코호트 축이 다름에 주의)
lm = pd.DataFrame([{"image_id": iid, "category_id": cat,
                    **cdqc.extract_mask_image(m, images[iid], p)}
                   for (iid, cat), m in masks.items()])
# lm도 cohort_stats/apply_z 동일하게 사용 (이름 기반이라 코드 차이 없음)
```

### 평가 (오측정 라벨이 있을 때)

```python
r = cdqc.recall_at_fpr(scores[bad], scores[good], fpr=0.05)
#  → {"threshold", "recall", "fpr_actual", ...}
hit = cdqc.localization_rate(top_k_preds, true_bad_sets, k=1)
tab = cdqc.ablation_table(z, {"기하만": [...], "+이미지증거": [...]}, labels)
```

### 16-bit 원본 tif

```python
img8 = cdqc.to_uint8(img16)             # percentile 스트레치 (기본) 또는 method="shift"
```

**주의**: 이미지별 percentile 스트레치는 절대 밝기를 없앤다 — `dyn_range`/`sat_*`
같은 L1 피쳐는 스트레치 **전** 원본에서 의미가 있으므로, 16-bit 원본을 쓸 경우
L1은 변환 전 이미지로 따로 계산하는 것을 권장.

## API 요약

| 함수 | 역할 |
|---|---|
| `extract_l3(img, S, E, px_nm, value_nm=None, params)` | 시퀀스 하나 → CD별 피쳐 (delta/cnr/rise/margin/npk/overshoot/plateau/pol/edge_valid + cd_nm/잔차/dstep/obliquity/value_mismatch) |
| `extract_l2(l3_df, group_cols, params)` | 시퀀스별 요약 (n_cd, cd_median/mad, delta_median, traj_rms + 총체 실패 축: angle_median(원형)/angle_spread/pitch_median/span_nm) |
| `extract_l1(img, params)` | 이미지 전역 피쳐 (noise/sat/dyn_range/struct_energy/tile_cv + hist) |
| `extract_mask_l3(mask, img, S, E, px_nm, params)` | CD별 마스크 정합 (mdist/mgrad/minside) — extract_l3와 index join |
| `extract_mask_image(mask, img, params)` | 마스크 형태 이상 (grad_agree/성분/구멍/거칠기/면적) — level "lm" |
| `cohort_stats(df, params)` | 받은 행 전체 = 한 코호트. 피쳐별 robust 통계 (JSON 가능 dict) |
| `apply_z(df, stats, params)` | directed z (`z_*`) + 부호 z (`zs_*`, both 피쳐) 추가 |
| `top_feature(z_df)` | 행별 최대 z·피쳐·사유코드 (curv는 기본 제외) |
| `threshold_from_quantile(values, q)` | 분위수 임계값 계산 |
| `impact_nm(cd_nm, flags, stat)` | 플래그 제외 시 통계량 변화 (nm) |
| `max_run(flags)` / `hist_emd(h, t)` / `robust_stats(x)` | 판정·비교 연산 |
| `recall_at_fpr` / `localization_hit·rate` / `ablation_table` | 평가 헬퍼 (라벨은 사용자가) |
| `to_nm` / `normalize_unit` / `infer_px_nm` / `ratio_cv` / `to_uint8` | 단위·px_nm·비트깊이 유틸 |
| `transform_coords` / `convention_scores` | 좌표 컨벤션 변환·후보 점수표 |
| `Params` / `FEATURES` | 연산 파라미터 dataclass / 피쳐 메타데이터 registry |

## 개발 검증 (사외 전용 — 사내 사용과 무관)

```
python -m cdqc.selftest      # 합성 데이터 감도/특이도 39셀 — 전부 PASS여야 함
python -m pytest tests -q    # 단위 테스트
```

selftest는 공개 API만 사용해 조립돼 있어(사내 사용자 코드와 같은 모양) API
자체의 검증이기도 하다. 합성 PASS는 "기계적으로 맞다"는 뜻이지 실제 TEM
실패를 잡는다는 증명이 아니다 — 유효성은 실데이터 라벨로만 증명한다.
