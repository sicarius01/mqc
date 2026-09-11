# cdqc — TEM CD 측정 품질 판정용 연산 라이브러리

**v0.6.1: 폴더 기반 진단 GUI와 NASCA Excel 로더.** 루트 디렉토리와 파일 매칭 규칙을 저장하면
하위 폴더에서 DM3·TIF·세그멘테이션 PNG·측정 XLSX를 찾아 일괄 분석한다.
이미지 오버레이, CD별 피쳐·점수·프로파일, 분포, 정상 기준 통계, 입력 문제를 화면에서 확인한다.
설치·실행·규칙 설정은 [GUI 사용 안내](GUI_GUIDE.md)를 참조한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\start_gui.ps1
```

GUI는 `cdqc_workbench` 패키지에 있고, 아래 `cdqc` 라이브러리의 순수 연산 API는 유지한다.
NASCA XLSX 읽기는 `cdqc.func.read_nasca_csv`가 Windows에 설치된 Excel을
`win32com.client.DispatchEx("Excel.Application")`로 별도 실행해 처리한다.
창은 기본적으로 숨기고, 읽은 값을 DataFrame으로 반환한 뒤 통합문서와 해당 Excel을 종료한다.
XLSX 입력에는 **Windows용 Microsoft Excel과 pywin32**가 필요하며, pandas/openpyxl 직접 읽기로 대체하지 않는다.

DL 세그멘테이션 레시피가 뽑은 CD 측정값 `(sx, sy, ex, ey)`의 품질 판정에
필요한 **연산만** 제공하는 순수 함수 라이브러리다. 피쳐 추출(CD/시퀀스/이미지),
robust 통계, directed z, 판정 헬퍼까지 — 그 이상은 하지 않는다.

- **파일을 읽지 않는다.** CSV/이미지 로딩, 컬럼 정리는 전부 사용자 코드.
- **워크플로우가 없다.** CLI·config 파일·출력 규약 없음. 함수 호출 → 리턴.
- **기준을 정하지 않는다.** 코호트 분리·임계값·공차·판정 정책은 사용자 몫.

설계 문서: `cdqc_spec.md` (피쳐 의미·정규화 원리), `cdqc_change_02_library_api.md`.

**현재 단계는 피쳐 추출 파이프라인을 완성하는 것이다.** 판정 로직(임계값, 피쳐
취사선택, 카테고리별 정책)은 여러 레시피·구조의 데이터를 누적하며 다듬을
영역이라 지금 확정하지 않는다. 그래서 라이브러리는 **전부 계산하고 전부
기록하며**, 피쳐를 빼거나 무력화하는 자동 규칙을 두지 않는다.

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
- 확인: `python -c "import cdqc; print(cdqc.__version__)"` → `0.6.1`

## 입력 계약

| 항목 | 형식 |
|---|---|
| 좌표 `S`, `E` | `(n,2)` float ndarray, **(x=col, y=row), zero-origin, 이미지 픽셀** |
| 시퀀스 | 한 번의 `extract_l3` 호출 = 한 (이미지 × CD 카테고리). **측정 순서대로 정렬**해서 넣을 것 (시퀀스 피쳐가 순서에 의존) |
| 이미지 | `np.ndarray[uint8]`, shape (H, W). `None`이면 기하 피쳐만 |
| 길이 단위 | 함수 경계에서 항상 **nm** (`to_nm`은 Å/pm/nm/µm/mm/m 인식, 모르는 단위는 즉시 에러) |
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

# ── 3) 정규화: 코호트 축은 사용자가 지정 (예: 카테고리별) ─────────────
is_normal = ...                          # 정상으로 간주할 행 — 사용자 판단
z = cdqc.cohort_z(l3, ["category_id"], base_mask=is_normal, params=p)
#   그룹별 robust z + pooled MAD 폴백(MAD=0 폭발 방어) + z 상한(30)
#   → z_*(directed) + zs_*(부호, both 피쳐) 컬럼 추가
agg = cdqc.aggregate_z(z)                # z_max / z_top{k}_kth / z_top{k}_mean
                                         #   + n_z_valid

# ── 4) 판정: 기준은 사용자, 연산은 헬퍼 ──────────────────────────────
top = cdqc.top_feature(z)                          # 행별 최대 z + 피쳐 + 사유코드
t = cdqc.threshold_from_quantile(agg.loc[is_normal, "z_top3_kth"], 0.99)
abs_f = cdqc.abs_flags(l3, {"bdist_s": 1.0, "obliquity": 5.0})   # 임계는 사용자가
flag = (agg["z_top3_kth"] > t) | (abs_f["n_abs_flags"] > 0)

# n_cd처럼 이산·규칙적인 값은 z가 아니라 최빈값 불일치로 본다
mode_f = cdqc.mode_flags(l2, ["n_cd"], ["category_id"])

roll = cdqc.flag_rollup(l3, flag, ["image_id", "category_id"])    # frac/max_run
impact = cdqc.impact_nm(seq["cd_nm"], seq_flags)   # 공차(nm)와 직접 비교
```

**정규화에서 세 가지를 조심한다.**

- `cohort_stats` + `apply_z`는 **한 코호트**용 원시 연산이다. 카테고리별로 돌면
  MAD가 0인 카테고리에서 z가 폭발한다 (이산 피쳐 `npk`는 거의 항상 1,
  `bdist_s`는 정상이면 정확히 0.000). `cohort_z`가 pooled MAD 폴백과 상한을
  얹어준다. 고정 통계를 저장했다 재사용할 때는 `floor_stats(stats, pooled, p)`.
- **집계는 두 정의를 다 낸다.** `z_top{k}_kth`는 k번째로 큰 z라 "k개 피쳐가
  동시에 넘었다"를 뜻하고(다중비교 방어), `z_top{k}_mean`은 상위 k개 평균이다.
  kth는 반응 피쳐가 k개 미만인 실패를 놓치고(예: CD 하나를 자기 중점 기준으로
  회전 → 2피쳐 신호), mean은 하나가 상한까지 포화하면 여전히 커서 방어가
  온전하지 않다. **전부 계산해 두고 판정 단계에서 고른다.**
- **z 크기를 심각도로 읽지 말 것.** 30은 포화값이고, 이동량이 커지면 오히려
  탐색 창을 벗어나 z가 떨어진다. 탐지 여부만 의미가 있다.
- **이산·규칙적인 값에는 z를 쓰지 말 것.** `n_cd`는 카테고리 안에서 늘 같아
  MAD가 0이고, 그러면 "CD 1개 누락 = z 2.0"처럼 `mad_floors`가 민감도를
  결정해버린다. `mode_flags`가 맞다.

### 세그멘테이션 흐름 — 라벨맵 경로가 권장 경로다

세그멘테이션 PNG를 읽는 것까지가 사용자 몫이다 (해상도가 이미지와 다르면
E-ARG-07 — 리사이즈도 명시적으로 사용자가).

```python
seg = cv2.imread(seg_path)                          # 사용자: 파일 읽기 (BGR)
seg = cdqc.close_annotation(seg)                    # CD 컬러 선이 라벨을 덮었으면 복구
labelmap = seg[..., 1]                              # 단일 채널 라벨맵

maps = cdqc.mask_maps(labelmap, images[iid])        # ★ 이미지당 1회
for (iid, cat), g in df.groupby(["image_id", "category_id"], sort=False):
    f = cdqc.extract_l3(images[iid], S, E, px, params=p)
    fb = cdqc.boundary_features(maps, S, E, px, params=p)
    f = pd.concat([f, fb], axis=1)      # 행 순서가 같아 index join
    #  → bdist_s/e/mid, bdist_ratio, label_runs, bgrad_s/e, label_s/e/mid
```

`mask_maps`에 이미지를 같이 넘기면 `bgrad_s/e`(보고 좌표 **그 자리**의
그래디언트/국소 σ)까지 나온다 — 프로파일 기반인 `cnr`과 독립적인 증거라
둘이 어긋나면 그 자체가 정보다.

**왜 라벨맵 경로인가.** 경계를 *모든 라벨 전이의 합집합*으로 잡으므로 클래스를
고를 필요가 없다 — CD의 S와 E가 서로 다른 계면에 있어도 둘 다 측정되고,
거리맵을 이미지당 1회만 계산한다.

`bdist_ratio = max(bdist_s, bdist_e) / bdist_mid`는 비율이라 배율에 무관하고,
**cnr로는 구분되지 않던 실패**를 잡는다: 계면이 아니라 층 안 허공에 그어진
CD는 양 끝의 그래디언트 증거가 정상과 겹치지만, 정상 CD는 ≈0 / 허공선은 ≈1이다.

이진 마스크 경로(`extract_mask_l3`)는 유지하지만 권장하지 않는다 — 클래스를
골라야 하고, S/E가 다른 계면이면 측정할 수 없으며, 허공에 그은 선도 어느 층
안에는 있으므로 `minside`를 통과한다. 경계 위치의 이미지 증거(`mgrad`)가
필요할 때만 쓴다.

```python
# 마스크 형태 자체의 이상 (이미지×클래스당 한 행 — L1과 코호트 축이 다름에 주의)
lm = pd.DataFrame([{"image_id": iid, "category_id": cat,
                    **cdqc.extract_mask_image(m, images[iid], p)}
                   for (iid, cat), m in masks.items()])
# lm도 cohort_z 동일하게 사용 (이름 기반이라 코드 차이 없음)
```

### 검증·진단 (실데이터에서도 그대로 돈다)

```python
# 코호트 통계를 저장해 둔다 — 주입 케이스로 통계를 다시 내면 주입이 흡수된다
z, stats = cdqc.cohort_z(l3, ["category_id"], base_mask=is_normal,
                         params=p, return_stats=True)

# grad_sigma_px를 데이터가 정하게 한다 (이웃 σ끼리 답이 일치하는 최소 σ)
sel = cdqc.select_grad_sigma(images[iid], S, E, px, params=p)
#  → {"sigma", "agreement", "valid_frac", "converged"}  — 채택은 사용자

# 주입 테스트 — 콜러블 하나만 주면 합성이든 실데이터든 같은 코드로 돈다
tab = cdqc.injection_test(
    S, E,
    lambda s, e: cdqc.extract_l3(images[iid], s, e, px, params=p),  # 콜러블
    stats[("catA",)],
    cases=[("shift_one", 5.0), ("rotate_one", 2.0), ("rotate_frame", 1.0),
           ("all_shift", 5.0), ("drop_one", 0.0)],
    l2_stats=l2_stats[("catA",)], params=p)
#  → kind별 l3_<집계>… + l3_top_feature + l3_rise, l2_* — 무엇이 걸렸는지까지

# 카테고리별 진단 지표 — **기록만 한다.** 이 값으로 피쳐를 빼지 않는다
summ = cdqc.category_summary(l3, ["category_id"], agg, traj_by="image_id")
#  → linearity, monotonic, *_mad_px, cnr_med_*, bdist_ratio_med,
#    delta_B_med_*, n_z_valid_med, z_top3_kth_p99
```

궤적 진단은 `traj_by`로 트래젝토리를 쪼갠 뒤 중앙값을 낸다 — 여러 이미지의
좌표를 한 덩어리로 PCA에 넣으면 궤적이 겹쳐 `linearity`가 무의미해진다.

`all_shift`(시퀀스 전체 이동)와 `drop_one`(CD 누락)은 **CD 레벨로는 원리적으로
탐지할 수 없다** — 통째로 밀리면 이웃 대비 잔차가 전부 0이고, 없어진 CD는 z를
낼 행 자체가 없다. L2 요약을 같은 카테고리의 다른 이미지들과 비교해야 보인다.

회전은 두 종류를 따로 본다: `rotate_one`(CD 하나가 자기 중점에서 돌아감)은
끝점이 궤적 **진행 방향**으로 움직여 법선 잔차가 원리적으로 못 보므로
`obliquity`/`angle_resid_seq` 2피쳐 신호에 그치고, `rotate_frame`(기준 각도
오설정)은 모든 CD가 옮겨져 여러 피쳐가 함께 반응한다.

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
| `extract_l3(img, S, E, px_nm, value_nm=None, params)` | 시퀀스 하나 → CD별 피쳐 (delta/delta_B/delta_scatter/cnr/rise/margin/npk/overshoot/plateau/pol/edge_valid + cd_nm/잔차/dstep/obliquity/angle_resid/value_mismatch) |
| `extract_l2(l3_df, group_cols, params)` | 시퀀스별 요약 (n_cd, cd_median/mad, delta_median, traj_rms, bdist_median + 총체 실패 축: angle_median(원형)/angle_spread/pitch_median/span_nm) |
| `extract_l1(img, params)` | 이미지 전역 피쳐 (noise/sat/dyn_range/struct_energy/tile_cv + hist) |
| `mask_maps(labelmap, img=None)` | 라벨 전이 합집합 경계 + 거리맵(+ img를 주면 그래디언트/국소 σ) — **이미지당 1회** |
| `boundary_features(maps, S, E, px_nm, params)` | CD별 라벨 경계 정합 (bdist_s/e/mid, bdist_ratio, label_runs, bgrad_s/e, label_s/e/mid) — **권장 경로** |
| `extract_mask_l3(mask, img, S, E, px_nm, params)` | CD별 이진 마스크 정합 (mdist/mgrad/minside) — 클래스 하나만 봄 |
| `extract_mask_image(mask, img, params)` | 마스크 형태 이상 (grad_agree/성분/구멍/거칠기/면적) — level "lm" |
| `cohort_stats(df, params)` / `apply_z(df, stats, params)` | 한 코호트 robust 통계 (JSON 가능 dict) / directed z 부여 |
| `cohort_z(df, group_cols, base_mask, params, return_stats)` | 그룹별 z + 분모 하한 4중 + z 상한 — 보통 이걸 쓴다. `return_stats=True`면 쓴 통계도 반환 |
| `floor_stats(stats, pooled, params)` | 고정 통계에 같은 분모 하한 적용 (재사용 캘리브레이션용) |
| `aggregate_z(z_df)` / `agg_columns()` | 행별 `z_max`/`z_top{k}_kth`/`z_top{k}_mean` + `n_z_valid` / 그 컬럼 이름들 |
| `abs_flags(df, thresholds)` | 물리 단위 절대 임계 플래그 + `n_abs_flags` (**임계값은 필수 인자**) |
| `mode_flags(df, cols, group_cols)` | 그룹 최빈값 불일치 플래그 + `n_mode_flags` — `n_cd`처럼 z가 안 맞는 값에 |
| `top_feature(z_df)` | 행별 최대 z·피쳐·사유코드 (curv는 기본 제외) |
| `threshold_from_quantile(values, q)` | 분위수 임계값 계산 |
| `impact_nm(cd_nm, flags, stat)` / `flag_rollup(l3_df, flags, group_cols)` | 플래그 제외 시 통계량 변화 (nm) / 시퀀스 롤업 (frac_flagged, max_run) |
| `max_run(flags)` / `hist_emd(h, t)` / `robust_stats(x)` | 판정·비교 연산 |
| `recall_at_fpr` / `localization_hit·rate` / `ablation_table` | 평가 헬퍼 (라벨은 사용자가) |
| `inject_coords` / `injection_test` | 알려진 실패 주입 → 고정 코호트 기준 z 상승 측정 (`shift_one`/`rotate_one`/`rotate_frame`/`swap`/`all_shift`/`drop_one`) |
| `select_grad_sigma(img, S, E, px_nm)` | 이웃 σ끼리 답이 일치하는 최소 `grad_sigma_px` — 채택은 사용자 |
| `trajectory_check(pts)` / `category_summary(l3_df, group_cols, agg, traj_by)` | 궤적·카테고리 진단 — **기록만 한다** |
| `to_nm` / `normalize_unit` / `infer_px_nm` / `ratio_cv` / `to_uint8` | 단위(Å/pm/nm/µm/mm/m)·px_nm·비트깊이 유틸 |
| `close_annotation(bgr)` | 세그멘테이션 PNG에서 컬러 주석이 덮은 라벨 복구 |
| `transform_coords` / `convention_scores` | 좌표 컨벤션 변환·후보 점수표 |
| `Params` / `FEATURES` | 연산 파라미터 dataclass / 피쳐 메타데이터 registry |

## 개발 검증 (사외 전용 — 사내 사용과 무관)

```
python -m cdqc.selftest      # 합성 데이터 감도/특이도 59셀 — 전부 PASS여야 함
python -m pytest tests -q    # 단위 테스트
```

## 배치 스크립트 (`scripts/` — 사용자 코드 계층)

```
scripts/run_batch.py     파일 로딩(dm3/tif/seg png/xlsx) · 좌표 변환 · 배치 루프 · 출력
scripts/feature_hist.py  카테고리별 피쳐 분포 그림
```

라이브러리가 아니라 **사내에서 그대로 쓰거나 고쳐 쓰는 예시**다 (패키지에
포함되지 않는다). 파일 I/O와 임계값처럼 cdqc가 하지 않기로 한 것들이 여기 있다.
**여기에 피쳐 계산이나 정규화 로직을 다시 쓰지 말 것** — 그러면 검증된 기능이
저장소 밖으로 새어나간다. 새 계산이 필요하면 cdqc에 넣고 여기서는 호출만 한다.

```python
from run_batch import run_batch, injection_test
out = run_batch([(dm3, tif, seg_png, xlsx), ...], exclude=["가이드라인카테고리"])
inj = injection_test(DATASETS[0], out["l3_stats"], out["l2_stats"])
```

selftest는 공개 API만 사용해 조립돼 있어(사내 사용자 코드와 같은 모양) API
자체의 검증이기도 하다. 합성 PASS는 "기계적으로 맞다"는 뜻이지 실제 TEM
실패를 잡는다는 증명이 아니다 — 유효성은 실데이터 라벨로만 증명한다.
