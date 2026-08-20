# cdqc — TEM CD 측정 품질 판정용 연산 라이브러리 스펙

버전 0.3 · 이 문서는 합의된 설계의 단일 진실원(single source of truth)이다.
설계 논의가 바뀌면 이 문서를 먼저 고친다.
(0.1 → 0.2: 프레임워크 → 순수 연산 라이브러리 재구조화, `cdqc_change_02_library_api.md`.
0.2 → 0.3: 마스크 피쳐 + 총체적 실패 축 + 평가 헬퍼 + to_uint8, 변경 지시 #03)

---

## 0. 한 문단 요약

DL 세그멘테이션 기반 레시피가 TEM 이미지에서 뽑은 CD 측정값 `(sx, sy, ex, ey)`가 제대로 된 것인지, 이미지와 측정 데이터에서 **의미를 아는 피쳐**를 뽑아 판정한다. 무지성 ML은 쓰지 않는다. 피쳐는 3단계 계층(CD → 카테고리 시퀀스 → 이미지)으로 뽑고, 코호트 내 robust 통계와 시퀀스 내 국소 잔차로 정규화한다. cdqc는 이 **연산만** 순수 함수로 제공한다 — 데이터 로딩, 코호트 분리, 임계값·판정 기준, 운영은 전부 사용자(사내) 코드의 몫이다.

---

## 1. 구조 원칙 (변경 지시 #02로 확정)

| | 사용자 (사내) | cdqc (라이브러리) |
|---|---|---|
| CSV/이미지 읽기, 컬럼 정리 | O | **안 함** (단위/px_nm/컨벤션 유틸만 제공) |
| 피쳐 계산 (CD/시퀀스/이미지) | 호출만 | **O — 핵심** |
| robust 통계·directed z | 호출만 | **O — 핵심** |
| 코호트 정의 (어떤 행이 한 코호트인지) | O | 안 함 |
| 임계값·공차·판정 기준·운영·저장 | O | 안 함 (연산 헬퍼만) |

- 모든 공개 함수는 순수 함수: 파일 I/O 없음, config 파일 없음, 전역 상태 없음, 런타임 네트워크 호출 0.
- 파라미터는 `Params` dataclass 하나 (기본값 내장). 저장/로드는 사용자 자유.
- 개발(사외) 검증은 합성 데이터로만: 합성 생성기와 selftest는 **핵심 자산**이지만 사내 사용과 무관한 개발 전용 모듈이다.
- **합성 테스트의 정직한 한계**: 합성에서 전부 PASS여도 "기계적으로 맞다"이지 "실제 TEM 실패를 잡는다"가 아니다. 유효성은 실데이터 라벨로만 증명한다.
- 사내 설치: `pip install -e .` (사내 pip 사용 가능 확인됨). 의존성 4개 핀: numpy, scipy, pandas, opencv-python-headless.
- 예외는 코드 부착(`CdqcError`, E-ARG-xx) — 사용자가 코드만 전달해도 사외에서 진단 가능.

## 2. 입력 계약 (함수 인자)

- **좌표**: `(n,2)` float ndarray, **(x=col, y=row), zero-origin, 이미지 픽셀 단위**. `(sx,sy)`와 `(ex,ey)`는 **서로 다른 두 엣지 위의 점**이고 잇는 선분 길이가 CD — 선분은 엣지를 따라가는 게 아니라 엣지 사이를 가로지른다.
- **시퀀스**: `extract_l3` 호출 하나 = 한 (이미지 × CD 카테고리). 같은 카테고리의 CD들은 ROI를 슬라이딩하며 측정된 것이므로 `{S_i}`, `{E_i}`는 매끄러운 엣지 궤적(DL 컨투어의 샘플)이어야 하고, **측정 순서대로 정렬**해서 넣는다 (시퀀스 피쳐 전부가 순서에 의존).
- **이미지**: `np.ndarray[uint8]`, shape `(H, W)`. `None`이면 기하 피쳐만 계산 (이미지 증거는 NaN).
- **길이 단위**: 함수 경계에서 항상 **nm**.
- 좌표 컨벤션이 불확실하면 `convention_scores()`로 후보 8개 `{xy/rowcol}×{origin 0/1}×{y_flip}` 점수표를 얻는다 (점수 = 보고 좌표 ↔ 축방향 그래디언트 피크 거리 중앙값 px, 낮을수록 좋음. 참고 채택 기준: 1등 ≤ 0.75px AND 2등/1등 비 ≥ 2 — 판별력은 좌표 지터에 근본 제한). **판단은 사용자.**
- 단위 변환은 `to_nm(value, unit)` — Å 변형(U+00C5/U+212B/NFD/ASCII/이름)과 nm 계열 인식, **모르는 단위는 고유값 목록과 함께 E-ARG-03 즉시 중단 (추측 금지)**.
- px_nm이 없으면 `infer_px_nm(S, E, value_nm)` — 이미지 하나 분량의 value/기하 비 중앙값. **한계**: 이미지 전체가 균일하게 밀린 계통 편향은 역산에 흡수된다 (탐지는 `delta_median` 몫). 행별 비 산포는 `ratio_cv()`로 점검 (>1%면 좌표와 보고값이 따로 계산됐을 가능성).
- **DL segmentation 마스크** (있는 카테고리만): **이진 ndarray** (bool 또는 uint8 0/비0), 이미지와 같은 shape. 라벨맵 → 클래스별 이진 분리(컬러/라벨 PNG 해석)는 사용자 몫. 해상도가 이미지와 다르면 `E-ARG-07` — 리사이즈도 사용자가 명시적으로.
- 원본이 uint8이 아니면 `to_uint8(img)` (percentile 스트레치 또는 uint16 상위 8비트). **주의**: 이미지별 스트레치는 절대 밝기를 없앤다 — `dyn_range`/`sat_*` L1 피쳐는 스트레치 전 원본에서 계산 권장.

## 3. 피쳐 설계

### 3.1 원칙

1. **추출기는 완전 범용.** 레시피가 뭔지 모른다. 모든 CD에 동일한 고정 길이 벡터.
2. **의미 부여는 코호트가 한다.** 정상/이상은 코호트 분포(사용자가 자름)와 시퀀스 내 이웃이 결정.
3. **무차원화.** 세션 간 비교 가능하도록 노이즈·배경·이웃으로 나눈다.
4. **방향 메타데이터.** 피쳐마다 `worse_when = "low" | "high" | "both"` (registry에 내장, `Params.direction_overrides`로 오버라이드).

### 3.2 L3 — 개별 CD 피쳐 (`extract_l3`)

프로파일 샘플링: 세그먼트 축을 따라 `S − margin_eff` ~ `E + margin_eff`를 subpixel 샘플링 (`map_coordinates`, order=1). `margin_eff = max(margin_px, W/2 + 2)` — 바깥 여유가 탐색 창 절반보다 짧으면 창 크기만큼의 엣지 점프를 delta가 추적하지 못한다. 노이즈 억제를 위해 **국소 엣지 접선 방향**으로 `±ribbon_half_w` px 오프셋한 줄들을 평균(리본). 접선은 세그먼트 수직이 아니라 이웃으로 추정: `tangent_S(i) = normalize(S_{i+1} − S_{i-1})`. 창 크기 `W = clamp(win_frac·CD, win_min_px, win_max_px)`.

그래디언트 피크의 subpixel 정제는 포물선 보간이 아니라 **반치폭(≥0.5·peak) 구간 가중 센트로이드** — argmax 계열은 강도 의존 노이즈에서 밝은 쪽으로 계통 편향된다 (defocus로 피크가 넓고 낮을 때 두드러짐).

국소 노이즈 σ: 세그먼트 주변 패치의 고주파(Immerkær 커널) 잔차 MAD × 1.4826. 전역 σ 쓰지 않음 (TEM은 두께 편차로 전역이 무의미). 리본 평균만큼(1/√n) 보정.

**이미지 증거 (엔드포인트 S, E 각각)**

| 피쳐 | 정의 | worse_when | 반응해야 하는 주입 실패 |
|---|---|---|---|
| `delta_s`, `delta_e` | 보고 좌표 ↔ 그래디언트 피크(센트로이드 subpixel) 부호 있는 거리, nm. **1순위** | both | `edge_jump`, `systematic_bias`(중앙값으로) |
| `cnr_s`, `cnr_e` | 스텝 높이 / 국소 노이즈 σ | low | `low_contrast`, `noise_up` |
| `rise_s`, `rise_e` | 10–90% 상승폭, nm | high | `defocus` |
| `margin_s`, `margin_e` | 1등 피크 / 2등 피크 (1등 ±peak_suppress 샘플 억제 후) | low | `double_edge` |
| `npk_s`, `npk_e` | `g > npk_ratio·peak`인 유의 피크 개수 | high | `double_edge` |
| `overshoot_s`, `overshoot_e` | 스텝 양쪽 오버슈트 진폭 / 스텝 높이 (프레넬 프린지 대리) | high | `defocus` (fringe 옵션) |
| `plateau_cv` | S–E 사이 플래토 robust 변동계수 | high | `plateau_defect` |
| `pol_s`, `pol_e` | 스텝 극성 ±1 | 코호트 최빈값과 불일치 | `edge_jump`(옆 레이어로) |
| `edge_valid_s/e` | 피크가 탐색 창 경계에 걸리지 않음 (bool) | false | — |

**기하 / 시퀀스 잔차** (전부 **국소 추세** 대비 — 전역 중앙값 쓰지 말 것: 실제 테이퍼에서 전부 플래그됨)

| 피쳐 | 정의 | worse_when | 반응 |
|---|---|---|---|
| `cd_nm` | `hypot(ex−sx, ey−sy) × px_nm` | — (코호트) | `edge_jump`. oblique 과대(1/cosθ−1)는 코호트 CD 산포에 묻힐 수 있음 — oblique 검출은 `cd_resid`/`obliquity` 몫 |
| `cd_resid` | 국소 robust 추세 대비 CD 잔차, nm | both | `edge_jump`, `oblique` |
| `s_resid`, `e_resid` | 각 궤적의 국소 적합 대비 법선 방향 잔차, nm | both | `edge_jump` (한쪽만) |
| `dstep_s`, `dstep_e` | 이웃 간 1차 차분의 **법선 성분** 크기, nm (유클리드 크기는 측정 피치를 재는 것. 법선은 점프에 강건한 접선(롤링 메디안 창 7)에서) | high | `edge_jump` |
| `obliquity` | 세그먼트-엣지접선 각의 시퀀스 중앙값 대비 편차, deg | high | `oblique` |
| `curv_s`, `curv_e` | 궤적 3점 국소 곡률. **기본 비활성** — 노이즈와 진짜 곡률 구분 불가, `s_resid`가 대체. 계산·리포트용으로만 | high | — |
| `angle` | 세그먼트 절대 각도 | — (코호트) | — |
| `value_mismatch_nm` | 보고 측정값 − 기하 길이 × px_nm, nm (value_nm 인자를 준 경우) | both | 좌표·값 불일치 |

국소 추세: `Params.method` = `robust_linear`(국소 Theil–Sen, **기본**) 또는 `hampel`(러닝 메디안). hampel은 기울거나 휜 궤적에 계통 잔차를 남긴다 (베이스라인 z_s_resid p90: hampel 9.8 vs robust_linear 1.6). 시퀀스가 `min_seq_len` 미만이면 잔차는 NaN.

**마스크 정합 (extract_mask_l3 — 마스크가 있는 카테고리만, extract_l3와 index join)**

| 피쳐 | 정의 | worse_when | reason |
|---|---|---|---|
| `mdist_s`, `mdist_e` | 보고 좌표에서 가장 가까운 마스크 경계까지 거리(nm). 경계 distance transform을 좌표 위치에서 bilinear 샘플 | high | POSITION_MISMATCH |
| `mgrad_s`, `mgrad_e` | 최근접 마스크 경계점 위치의 **이미지** 그래디언트 크기 / 국소 노이즈 σ — DL 경계가 실제 명암 전이 위에 있는지 | low | POSITION_MISMATCH |
| `minside` | 세그먼트 중점(플래토 중앙)이 마스크 내부인가 (bool) — CD가 마스크 구조를 실제로 가로지르는지 | bool | GEOMETRY_ODD |

### 3.3 L2 — 카테고리 시퀀스 피쳐 (`extract_l2`)

| 피쳐 | 정의 |
|---|---|
| `n_cd` | CD 개수. 코호트 대비 (missing 감지) |
| `cd_median`, `cd_mad` | 코호트 대비 |
| `delta_median_s/e` | 시퀀스 전체 계통 편향. 개별 delta는 정상인데 전부 같은 방향이면 DL 편향 |
| `traj_rms_s/e` | 궤적 잔차 RMS |

**총체적 실패 축** (변경 #03 §2 — 기준 각도 오설정 류는 개별 CD z로 희석되므로 시퀀스 요약을 코호트와 비교):

| 피쳐 | 정의 | worse_when | 잡는 것 |
|---|---|---|---|
| `angle_median` | 세그먼트 각도의 **원형 중앙값** (deg, 180° 주기 — `atan2(median sin2θ, median cos2θ)/2`. 단순 median은 ±90° 경계에서 깨짐. 코호트가 랩 경계 근처 레시피면 주의) | both | 기준 각도 오설정 — 시퀀스 전체 회전 |
| `angle_spread` | 각도의 원형 MAD (deg) | high | 방향이 뒤죽박죽인 시퀀스 |
| `pitch_median` | 이웃 CD 중점 간 거리 중앙값 (nm) | both | 측정 간격 설정 오류 |
| `span_nm` | 첫/끝 CD 중점 간 거리 (nm) | both | ROI 길이가 다름 |

(`n_cd`가 이 류의 1차 신호. pitch/span은 extract_l3가 실어주는 캐리어 컬럼 mid_x/mid_y/px_nm에서 계산.)

**LM — 마스크 이미지 레벨 (extract_mask_image)**: 마스크는 보통 (이미지 × 클래스)당 하나라 **L1과 코호트 축이 다를 수 있다** — 코호트 분리는 사용자 몫 (cohort_stats/apply_z는 이름 기반이라 코드 차이 없음).

| 피쳐 | 정의 | worse_when |
|---|---|---|
| `mask_grad_agree` | 마스크 경계 픽셀 전체의 이미지 그래디언트 중앙값 / 노이즈 σ — **DL이 헛것을 그렸는지의 단일 지표** | low |
| `mask_n_components` | 연결 성분 수 (8-이웃) | both |
| `mask_hole_frac` | 성분 내부 구멍 픽셀 비율 | high |
| `mask_boundary_rough` | 경계 둘레 / 등면적 원 둘레 (**16px 이상** 성분별 중앙값 — 잡티가 중앙값을 지배하지 않게. 잡티는 n_components 몫) | high |
| `mask_area_frac` | 마스크 픽셀 비율 | both |

플래그 의존 값은 사용자가 플래그를 정한 뒤 헬퍼로 직접 계산한다:
`impact_nm(cd_nm, flags)` = |stat(전체) − stat(플래그 제외)| — **물리 단위(nm)라 스펙 공차와 직접 비교**. `max_run(flags)` = 연속 플래그 최대 길이 (뭉침=국소 이미지 손상 vs 산발=락온 실패 구분).

### 3.4 L1 — 이미지 피쳐 (`extract_l1`, 측정 무관 OOD)

| 피쳐 | 정의 |
|---|---|
| `noise_sigma` | Laplacian 응답 MAD × 1.4826 (커널 norm 정규화) |
| `sat_lo`, `sat_hi` | `mean(img==0)`, `mean(img==255)` |
| `dyn_range` | p99 − p1 |
| `struct_energy` | 그래디언트 크기 p90 / noise_sigma |
| `tile_energy_cv` | 4×4 타일별 struct_energy의 CV (부분 손상) |
| `hist` | 256-bin 정규화 히스토그램 — 코호트 템플릿과 `hist_emd()`로 비교 (템플릿은 사용자가 정상 이미지 hist들의 중앙값으로) |

FFT 전역 샤프니스는 넣지 않는다 (TEM에서 구조 변화와 초점 변화가 분리 안 됨).

## 4. 정규화와 판정 연산

### 4.1 두 층의 정규화 — 역할이 다름

| | 잡는 것 | 못 잡는 것 |
|---|---|---|
| **시퀀스 내 잔차** (같은 이미지·카테고리 이웃) — L3 피쳐에 내장 | 국소 이상, 점프. 세션 변동에 완전 면역 | 시퀀스 전체가 균일하게 밀린 것 |
| **코호트 대비** (`cohort_stats` + `apply_z`) | 전체 편향, 이미지 단위 이상 | 국소 이상 (희석됨) |

### 4.2 코호트 통계 (`cohort_stats`)

- **코호트 분리는 사용자가 한다** — 함수는 받은 행 전체를 한 코호트로 취급 (권장: recipe × category. 표본 부족 시 상위로 묶는 것도 사용자 판단).
- robust z: `(x − median) / max(1.4826·MAD, mad_floor)`. `mad_floor`는 `Params.mad_floors` (이산/퇴화 피쳐 MAD=0 방어: npk/n_cd/sat/noise_sigma/dyn_range/delta_median/value_mismatch 등 기본값 내장).
- 양수 heavy-tail 피쳐(`rise`, `margin`, `dstep`, `cd_mad`)는 log 후 z (`Params.log_features`).
- 오염 처리: 1차 z 상위 `trim_frac`(기본 5%) 제거 후 재계산 1회. **한계**: 불량률이 높은 데이터로 통계를 내면 오염된다 — 정상으로 확인된 행만 넣는 것을 권장.
- 반환은 JSON 직렬화 가능한 평범한 dict — 저장/로드(재사용 캘리브레이션)는 사용자 자유.

### 4.3 directed z (`apply_z`)

- z 계열: 방향 반영 — low → −z, high → +z, both → |z| (both는 부호 z `zs_*`도 보존: 계통 편향 분석용).
- bool 계열(edge_valid): False → 고정 z = `Z_ON_BAD`(4.0).
- match 계열(pol): 코호트 최빈값과 불일치 → 고정 z = `Z_ON_BAD`.

### 4.4 판정 (참고 패턴 — 기준·정책은 전부 사용자)

두 층의 임계 철학: **CD 레벨은 느슨**(오경보 비용 쌈 — 사람이 그 CD 하나 보면 끝), **이미지 레벨은 엄격**(재측정/재촬영 비용). 참고 구성:

- CD 플래그: `top_feature(z)`의 top_z > t_soft. 사유 코드 5종은 registry가 피쳐별로 제공 (`EVIDENCE_WEAK` / `POSITION_MISMATCH` / `SEQUENCE_JUMP` / `GEOMETRY_ODD` / `FOCUS_BLUR`).
- 이미지 레벨 세 갈래 OR: G0 = L1 z 단측 초과(이미지 자체 불량), G1 = `impact_nm > 공차[카테고리]`(보고 통계량 오염 — 공차는 필수 사용자 입력), G2 = 시퀀스 요약(delta_median, cd_median, n_cd) z 초과(계통 실패 — 개별 플래그 0개여도 잡힘).
- 임계값 계산 유틸: `threshold_from_quantile(정상 z, q)`. **분위수 기반이므로 피쳐 셋을 바꾸면 재계산 필수**, 정상 데이터로만 계산할 것.
- CD 레벨 임계를 느슨하게 잡아도 무해한 CD 몇 개 더 걸리는 건 impact를 안 움직인다 — 이게 두 층을 분리하는 이유.

클러스터링은 판정이 아니라 실패 모드 발견용 — 사용자가 플래그 CD의 z 벡터로 직접 (cdqc 범위 밖).

## 5. 파라미터 — `Params` dataclass

config 파일 없음. 필드: 샘플링(ribbon_half_w=2, win_frac=0.4, win_min/max_px=8/64, margin_px=6, grad_sigma_px=1.0, noise_patch_px=15, peak_suppress=6, npk_ratio=0.3), 시퀀스(local_window=9, method="robust_linear", min_seq_len=5), 통계(trim_frac=0.05, mad_floor_default=1e-6, mad_floors, log_features, direction_overrides). 기본값 근거는 §3–4.

## 6. 공개 API

`import cdqc` 로 전부 접근. §3–4의 함수들(`extract_l1/l2/l3`, `extract_mask_l3`, `extract_mask_image`, `cohort_stats`, `apply_z`, 판정 헬퍼) + **평가 헬퍼**(`recall_at_fpr`, `localization_hit/rate`, `ablation_table` — 라벨은 사용자가 만들고 cdqc는 연산만) + 유틸(`to_nm`, `normalize_unit`, `infer_px_nm`, `ratio_cv`, `to_uint8`, `transform_coords`, `convention_scores`) + 메타(`FEATURES` registry, `Params`, `CdqcError`). 시그니처와 사용 예제는 README.md가 기준.

## 7. 합성 데이터 생성기 + selftest (개발 전용)

### 7.1 이미지 모델

```
img = background(gradient) + Σ bands(erf edge profile, contrast, position)
      + fringe(옵션) + noise(Poisson-like) → clip → uint8
```

- 수직/약간 기울고 휜 밴드들. 밴드 경계가 엣지, 카테고리 = (좌, 우) 엣지 쌍.
- "레시피 출력" = 참 좌표 + 지터(`coord_jitter_px=0.3` — DL 컨투어 현실치). `value` = **참 길이 × px_nm** (주입 후에도 유지 → 좌표·값 불일치 경로 검증), 단위는 행마다 Å/nm 교대.
- 전부 메모리 내 (`generate_dataset() → (records, images)`), 저장은 눈 확인용 옵션.

### 7.2 주입 실패 (독립 강도 조절)

| 실패 | 구현 | 반응해야 할 피쳐 | 반응하면 안 되는 피쳐 (특이도) |
|---|---|---|---|
| `defocus` | 엣지 rise 배수 ↑ (+fringe 옵션) | `rise`, `overshoot`(fringe 시) | `delta`, `cd_resid` |
| `low_contrast` | 층 대비 배수 ↓ | `cnr` | `rise`, `delta` |
| `noise_up` | 노이즈 σ 배수 ↑ | `cnr`, L1 `noise_sigma` | `delta_median` |
| `edge_jump` | 일부 CD의 S를 d nm 이동 (창 안 강도) | `delta_s`, `s_resid`, `dstep_s`, `cd_resid` | `e_*` |
| `systematic_bias` | 시퀀스 전체 S를 d nm 이동 | `delta_median_s` | 개별 `s_resid`, `dstep_s` |
| `oblique` | 일부 세그먼트를 θ 회전 (양 끝 엣지 위 유지) | `obliquity`, `cd_resid` | `delta`, `cnr` |
| `missing` | 일부 CD 삭제 | `n_cd` | 나머지 CD 피쳐 |
| `double_edge` | 플래토 쪽에 약한 고스트 라인 | `margin`, `npk` | `cnr` |
| `plateau_defect` | S–E 사이 블롭 | `plateau_cv` | `delta` |
| `saturation` | 대비 스트레치 → 클립 | L1 `sat_lo` | — |
| `partial_damage` | 일부 타일 노이즈/블러 | L1 `tile_energy_cv` | 다른 타일 CD의 `delta` |
| `mask_shift` | 마스크만 d px 평행이동 (이미지·좌표 정상) | `mdist_*`, `mgrad_*`(경계가 플래토 위로), LM `mask_grad_agree` | `cnr_s`, `delta_s` |
| `mask_ragged` | 경계 안쪽 1px 플립(거칠기) + 바깥 거리-4 링 저밀도 잡티(성분 수 — 병합 없이 단조 증가하도록 분리 배치) | LM `mask_boundary_rough`, `mask_n_components` | `mdist_s` (중앙값 기준) |
| `rotated_frame` | 시퀀스 전체 θ 회전 배치 + CD 수 감소 (회전 세그먼트도 엣지 위) — 총체적 실패 | L2 `angle_median`, `n_cd` | 개별 `delta_s` |

### 7.3 selftest (`python -m cdqc.selftest`)

**공개 API만 사용해 조립** (사내 사용자 코드와 같은 모양 — API dogfooding). 베이스라인으로만 코호트 통계를 내고 주입 케이스를 그 기준으로 채점한다 (케이스 자체 코호트로 정규화하면 주입이 흡수됨).

- "반응해야 할" 칸: 강도 오름차순 단조 증가(슬랙 0.5) + 최종 z ≥ 3.
- "반응하면 안 되는" 칸: 전 강도 |z| < 1.5 — **계통 편향(부호 z) 기준**: both 피쳐의 directed z는 산포만 커져도 오르므로(예: defocus에서 delta 산포 증가) 부호 z 중앙값으로 잰다. 산포로 인한 개별 오플래그는 분위수 임계값이 관리.
- 전부 PASS면 exit 0. **재구조화·수정 후 이 39셀 PASS 재현이 회귀 기준.**

## 8. 코드 구조

```
cdqc/
  __init__.py            # 공개 API re-export (__all__)
  api.py                 # extract_l1/l2/l3, cohort_stats, apply_z, 판정 헬퍼
  params.py              # Params dataclass
  utils.py               # 단위/px_nm/좌표 컨벤션 유틸
  errors.py              # CdqcError, E-ARG-xx
  geometry.py            # 접선, 국소 추세 잔차, 법선 dstep, 곡률, obliquity
  sampling.py            # 리본 샘플러, 국소 노이즈
  evidence.py            # 프로파일 → delta/cnr/rise/margin/npk/overshoot/극성
  features/
    l3.py                # 시퀀스 피쳐 조립 (순수 함수)
    l1.py                # 이미지 피쳐
    mask.py              # 마스크 정합(CD 레벨) + 마스크 형태(lm 레벨)
    registry.py          # 피쳐 이름/worse_when/사유코드/설명 — 단일 목록
  synth/                 # 개발 전용: generator.py(SynthParams, 메모리 내), inject.py
  selftest.py            # 개발 전용: python -m cdqc.selftest
tests/                   # pytest
pyproject.toml           # 의존성 4개 핀, pip install -e .
README.md                # 사용 예제 (API 기준 문서)
```

코딩 규칙: 피쳐 함수는 순수 함수 · 피쳐 추가 = registry 한 줄 + 계산 함수 하나 (registry가 정규화 방향·사유코드·selftest를 구동) · 예외는 E-코드 · 시드 고정 재현 · 프로파일 샘플링은 CD 배치로 `map_coordinates` 호출 최소화 · docstring에 피쳐의 의미 한 줄.

## 9. 왕복 프로토콜 (사외 개발 ↔ 사내 실행)

```
사외: 패치 + git push (selftest 39셀 PASS 상태로만)
  ↓ 사내: git pull (editable 설치라 재설치 불필요)
사내: 사용자 스크립트 실행 → 무차원 요약(z 분포, 플래그율, 상관, 에러 코드) 검토
  ↓ (규정상 반출 가능한 것만) 무차원 통계/서술/에러 코드
사외: 해석 → 다음 패치
```

- 사외로 나오는 것: 무차원 통계, PASS/FAIL, 카운트, E-코드, 서술. **뭘 내보낼지는 사용자가 규정 보고 판단. 애매하면 안 보낸다.**
- 사내 조정은 Params/임계값 등 사용자 코드에서 — cdqc 수정 없이 실험 가능해야 왕복이 준다.

## 10. 미결 사항 (알게 되면 이 문서 갱신)

- ~~DL 마스크/확률맵 접근~~ → **해결 (변경 #03)**: segmentation PNG 확보, 마스크 피쳐 추가됨. 확률맵(soft mask)은 여전히 미확보 — 생기면 mgrad를 확률 가중으로 확장 검토.
- 카테고리별 공차(tolerance): 사용자가 스펙에서 가져와 impact와 비교. auto 없음.
- 레시피 최종 통계량 median/mean 여부: `impact_nm(stat=...)` 인자로 대응. 확인 필요.
- `local_window` 기본 9가 실데이터 시퀀스 길이 대비 적정한지: 결측률 보고 판단 (5~7 후보).
- 실데이터 라벨 확보 후 검증(재현율/국소화)은 사용자 스크립트로.
