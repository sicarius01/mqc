# cdqc — TEM CD 측정 품질 판정용 연산 라이브러리 스펙

버전 0.6 · 이 문서는 합의된 설계의 단일 진실원(single source of truth)이다.
설계 논의가 바뀌면 이 문서를 먼저 고친다.
(0.1 → 0.2: 프레임워크 → 순수 연산 라이브러리 재구조화, `cdqc_change_02_library_api.md`.
0.2 → 0.3: 마스크 피쳐 + 총체적 실패 축 + 평가 헬퍼 + to_uint8, 변경 지시 #03.
0.3 → 0.4: 실데이터에서 검증된 스크립트 기능 승격 — 라벨 경계 피쳐, σ 스윕
delta, 코호트 z 편의층(pooled 폴백·z 상한·상위 k 집계), 주입/궤적/카테고리
진단, 원형 각도 잔차, mask_grad_agree 국소 σ 수정.
0.4 → 0.5: 원본 스크립트 대조 반영 — 집계 두 정의(kth/mean) 병행, 회전 주입
2종 분리, `bgrad`/`label_*` 복원, 분모 하한 4중, `mode_flags`, σ 자동 선택
승격, 단위 m/mm/µm/pm 인식, 궤적 진단을 트래젝토리별로.
0.5 → 0.6: 이미지 미측정의 z 분리, L2 각도 원형 정규화, 결측 코호트 키의
명시적 입력 오류. 별도 `cdqc_workbench` 계층에 로컬 폴더 기반 진단 GUI 추가.)

### v0.6 입력·통계 및 GUI 계약

- v0.6.1: NASCA XLSX 입력은 `cdqc.func.read_nasca_csv(path, visible=False, header=True)`를 사용한다.
  이 함수만 명시적인 파일 I/O 어댑터로 제공하며, 핵심 피쳐·정규화 API의 순수 연산 계약은 유지한다.
  `DispatchEx("Excel.Application")`로 별도 Excel을 창 없이 실행해 원본 경로를 읽고 DataFrame을 반환한다.
  GUI와 배치 스크립트의 XLSX 로딩은 이 경로로 통일한다.

- `extract_l3(img=None)`의 `edge_valid_s/e`는 미측정 NaN이다. `apply_z`는
  결측 bool/match 피쳐를 NaN으로 유지하고 실제 False만 고정 이상 점수로 계산한다.
- L2 `angle_median`의 코호트 중심/MAD와 편차는 180도 주기로 계산한다.
  상대 스케일 하한은 원형 변수의 임의 원점에 의존하지 않는다. 고정 기준에도 같은 계산을 쓴다.
- `cohort_z`는 결측 그룹 키를 `E-ARG-06`으로 거절하며 컬럼별 결측 건수를 알린다.
- 파일 탐색·설정 저장·시각화는 `cdqc_workbench`가 담당한다. `cdqc`에 파일 I/O를 넣지 않는다.
  4종 파일의 하위 디렉토리·확장자·접미어를 독립 설정하고, 측정 기준 폴더와 공통 파일명으로 묶는다.
  초기값은 DM3/TIF가 기준 폴더, PNG/XLSX가 `result`, XLSX 접미어가 `_Result`다.
  입력 누락·중복·계산 제외 행은 GUI에 표시한다. 정상 기준과 평가 데이터를 분리해 재사용할 수 있다.
  실행과 관찰 방법은 [GUI 사용 안내](GUI_GUIDE.md)를 참조한다.

**현재 단계는 피쳐 추출 파이프라인을 완성하는 것이다.** 판정 로직(임계값,
피쳐 취사선택, 카테고리별 정책)은 여러 레시피·구조의 데이터를 누적하면서
계속 다듬을 영역이며 지금 확정할 수 없다. 따라서 이 버전의 원칙은
**"전부 계산하고 전부 기록한다"** — 피쳐를 빼거나 무력화하는 자동 규칙을
만들지 않는다 (§11).

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
- 단위 변환은 `to_nm(value, unit)` — **Å/pm/nm/µm/mm/m** 인식 (Å은 U+00C5/U+212B/NFD/ASCII/이름, µ는 U+00B5/U+03BC 모두). SI 접두어는 모호하지 않으므로 추측이 아니다. 실데이터 xlsx의 `MeasurementUnit`이 `"m"`인 경우가 있어 미터 계열이 필요하다. **모르는 단위는 고유값 목록과 함께 E-ARG-03 즉시 중단 (추측 금지)**.
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

**σ 스윕 (`delta_B_*`, `delta_scatter_*`)**: `delta` 하나는 `grad_sigma_px` 선택에 의존한다. σ를 `delta_sigma_sweep`(기본 0.25~1.5, 6개)로 훑어 반복 측정하고 **중앙값을 `delta_B_s/e`(px), MAD를 `delta_scatter_s/e`(px)** 로 쓴다. 파라미터 의존성이 사라지고, scatter는 그 자체로 강력한 피쳐가 된다:

| delta | scatter | 상황 | 대응 |
|---|---|---|---|
| 크다 | 작다 | 엣지는 선명한데 좌표가 딴 데 | 측정 실패 → 재측정 |
| 크다 | 크다 | 엣지 자체가 없어 아무 데나 | 이미지 문제 → 재촬영 |

(엔드포인트별로 낸다 — `delta_s`/`delta_e`, `cnr_s`/`cnr_e` 등 다른 엔드포인트 피쳐와 같은 규약. 단위는 이름대로 **px**이고 `delta_s`/`delta_e`는 nm이다.)

**σ 자동 선택 (`select_grad_sigma`)**: 주 피쳐에 쓸 `grad_sigma_px` 자체도 사람이 정할 값이 아니다 (§11-3). σ를 훑으면서 **이웃 σ끼리 답이 일치하기 시작하는 최소 σ**를 고른다 — 일치도는 이웃 σ 간 delta 차이의 **p90**(중앙값은 절반만 맞아도 통과해서 너무 관대하다), 유효 delta 비율이 `min_valid` 미만인 σ는 후보에서 뺀다. 결과를 `Params(grad_sigma_px=…)`에 넣는 것은 사용자가 한다 — **코호트마다 다른 σ를 쓰면 delta 통계가 섞인다.**

**`sanitize` — 비율 피쳐의 분모 폭발 방어 (일반 규칙)**: 스텝이 노이즈 수준이면 `overshoot = 오버슈트 진폭 / 스텝 높이`는 0으로 나눈 값이고, 1px의 절반도 안 되는 `rise`는 물리적으로 측정 불가다. `cnr < min_cnr_for_ratio`(1.0)면 `overshoot` → NaN, `rise < min_rise_px`(0.5px)면 `rise`와 `overshoot` → NaN. **이것은 카테고리 예외 목록이 아니라 피쳐 자체의 일반 규칙이다** (§11-1) — 어느 카테고리든 같은 조건에서 같은 이유로 값이 무의미하다.

**각도는 z 대상이 아니다**: `angle`(절대 각도)은 ① 180° 주기 원형 변수라 뺄셈이 정의되지 않고(179°와 −179°가 358 차이), ② 같은 카테고리 CD는 각도가 거의 같아 MAD가 0.01° 수준이라 0.5° 차이가 z 50이 되며, ③ TEM은 시료 방향이 매번 달라 절대 각도 자체가 무의미하다. registry에서 `kind="raw"`(계산·기록만)로 두고, 정규화는 **원형 잔차** `((θ − ctr + 90) mod 180) − 90`로 한다. **변환이 필수인 경우**이지 "정보가 없어서 뺀" 것이 아니다.

중심을 어디로 잡느냐에 따라 잡는 실패가 달라져서 **둘 다 계산한다**:

| 피쳐 | 중심 | 시퀀스 전체 회전 | 계산 위치 |
|---|---|---|---|
| `angle_resid_seq` | 그 시퀀스의 원형 중앙값 | **0** (중심이 같이 돈다) → L2 `angle_median`이 담당 | `extract_l3` |
| `angle_resid_cohort` | 코호트(그룹)의 원형 중앙값 | 반응한다 | `cohort_z` |

층 분리는 `_seq` 쪽이 깔끔하지만, 어느 쪽이 실제로 유용한지는 데이터가 쌓여야 안다 (§11-2).

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

### 3.2b 라벨 경계 피쳐 (`mask_maps` + `boundary_features`) — 마스크 정합의 권장 경로

세그멘테이션 **라벨맵**(단일 채널 int 배열)의 **모든 라벨 전이의 합집합**을 경계로 삼고 `distance_transform_edt`로 거리맵을 만든다. 전이 양쪽 픽셀을 모두 경계로 표시한다 (참 계면은 두 픽셀 중심 사이라 어느 쪽을 골라도 ±0.5px 양자화가 남는다 — 양쪽 표시가 "경계 위 좌표 → 거리 0"을 보장한다). 이미지 프레임은 경계가 아니다.

클래스를 고르지 않기 때문에:
- **S와 E가 서로 다른 계면에 있어도** 둘 다 측정된다
- distance transform을 **이미지당 1회**만 돈다 (카테고리 × 클래스마다 돌던 방식 대비 대폭 감소)

| 피쳐 | 정의 | worse_when |
|---|---|---|
| `bdist_s`, `bdist_e` | 보고 좌표 ↔ 가장 가까운 라벨 전이 거리 (nm). 정상 계면 측정이면 ≈0 | high |
| `bdist_mid` | 세그먼트 중점 ↔ 라벨 전이 거리 (nm) — 층 반폭 대리, 비율의 분모 | both |
| `bdist_ratio` | `max(bdist_s, bdist_e) / bdist_mid` — **비율이라 배율 무관** | high |
| `label_runs` | 라벨 런 개수. **끝점에서 `label_inset_px` 안쪽 구간만** 센다 — 끝점은 계면 위라 라벨이 양쪽을 오가서, 끝까지 세면 정상 CD의 런 수가 측정마다 흔들린다 | both |
| `bgrad_s`, `bgrad_e` | 보고 좌표 **그 자리**의 이미지 그래디언트 / 국소 σ. `mask_maps`에 img를 준 경우에만 | low |
| `label_s`, `label_e`, `label_mid` | 각 지점의 라벨값 — 이 CD가 어느 층·어느 계면을 재는가 (kind="raw") | - |

`label_runs`의 방향이 `both`인 이유: 예상보다 많이 지나도(다중 관통) 적게 지나도(계면을 안 가로지름) 이상이다. 카테고리별 z이므로 그 카테고리의 정상 통과 횟수가 중앙값이 되고 양쪽 이탈이 잡힌다.

`bgrad`는 프로파일 기반인 `cnr`과 **독립적인** 증거다 (한쪽은 리본 평균 프로파일의 스텝 높이, 다른 쪽은 그 픽셀의 그래디언트). 둘이 어긋나면 그 자체가 정보다.

`bdist_ratio`는 **cnr로는 구분되지 않던 실패**를 잡는다: 계면이 아니라 층 안 허공에 그어진 CD는 양 끝의 그래디언트 증거가 정상 범위와 겹치지만(관찰된 예: cnr 7.96~9.13 대 정상 9~17), 정상 CD는 `bdist_ratio ≈ 0`, 허공선은 `≈ 1`이다.

초기 방식인 `extract_mask_l3`(이진 마스크 하나 = 클래스 하나 → mdist/mgrad/minside)는 **유지하되 권장하지 않는다**: 클래스를 골라야 하고, S/E가 다른 계면이면 측정 불가이며, 허공에 그은 선도 어느 층 안에는 있으므로 `minside`를 통과한다. 경계 위치의 이미지 증거(`mgrad`)가 필요할 때 쓴다.

세그멘테이션 PNG가 CD 컬러 주석에 덮여 라벨이 잘린 경우 `close_annotation(bgr)`으로 먼저 복구한다 (라벨값 자동 탐색 → 클래스별 closing → **주석이 있던 자리에서만** 채움).

### 3.3 L2 — 카테고리 시퀀스 피쳐 (`extract_l2`)

| 피쳐 | 정의 |
|---|---|
| `n_cd` | CD 개수. 코호트 대비 (missing 감지) |
| `cd_median`, `cd_mad` | 코호트 대비 |
| `delta_median_s/e` | 시퀀스 전체 계통 편향. 개별 delta는 정상인데 전부 같은 방향이면 DL 편향 |
| `traj_rms_s/e` | 궤적 잔차 RMS |
| `bdist_median_s/e` | 라벨 경계 거리 중앙값 (nm) — 시퀀스 전체가 계면에서 이탈 |

**L2가 존재하는 이유 (CD 레벨로는 원리적으로 불가능한 실패)**: 시퀀스가 통째로 밀리면 모든 CD가 똑같이 밀려 이웃 대비 잔차가 0이 되고, 없어진 CD는 행 자체가 없어 z를 낼 수 없다. 이미지 × 카테고리 요약값을 같은 카테고리의 **다른 이미지들**과 비교해야 보인다. 검증은 `injection_test`의 `all_shift`/`drop_one` 케이스가 한다 (§7.4).

플래그 롤업은 `flag_rollup(l3_df, flags, group_cols)` → `n_flagged`/`frac_flagged`/`max_run`. `max_run`은 게이트가 아니라 **사유 코드 결정용**이다 (뭉침 = 국소 이미지 손상, 산발 = 락온 실패).

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
| `mask_grad_agree` | 마스크 경계 픽셀의 (이미지 그래디언트 / **국소** σ) 중앙값 — **DL이 헛것을 그렸는지의 단일 지표**. 전역 σ를 쓰면 구조 대비에 물려 판별력이 사라지고(마스크를 0/5/20px 밀어도 0.57/0.38/0.41로 안 움직였다) 노이즈 없는 이미지에서 값이 폭주한다. 타일 σ 맵을 경계 픽셀 위치에서 조회하고, σ 하한은 uint8 양자화 수준(0.5), 값은 상한 클립 | low |
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

### 4.2b 코호트 z 편의층 (`cohort_z`, `aggregate_z`, `abs_flags`)

`cohort_stats` + `apply_z`는 **한 코호트**를 다루는 원시 연산이다. 실전에서 반복되는 세 가지를 `cdqc/normalize.py`가 얹는다. 코호트 축(`group_cols`)은 여전히 사용자가 정한다.

1. **그룹별 통계** — 카테고리마다 중심·스케일을 따로. 카테고리 고유 성질(예: cnr이 원래 1.5인 카테고리)이 자동 흡수된다. "원래 1.5인데 지금도 1.5"는 이상이 아니다. **카테고리 예외 목록을 만들지 않는 근거가 이것이다.**
2. **분모 하한 4중** (`floor_stats`) — 그룹 MAD가 죽으면 분모가 0에 붙어 z가 폭발한다. 첫 실행에서 `z_max` 99% 분위가 2,000,000이 나왔던 원인(이산 피쳐 `npk`는 거의 항상 1, `bdist_s`는 정확히 0.000)을 **피쳐를 빼지 않고** 막는다:

   ```
   분모 = max(1.4826·MAD_그룹,
             pooled_mad_frac · 1.4826·MAD_pooled,   ← 코호트 전체 산포 (0.30)
             rel_scale_floor · |중앙값|,             ← 상대 하한 (0.01)
             abs_scale_floor,                        ← 절대 바닥 (1e-3)
             mad_floors[피쳐])                       ← 피쳐별 물리 분해능
   ```

   `rel_scale_floor`가 없으면 `cd_nm`처럼 값이 수십 nm인 피쳐가 MAD만 작을 때 폭발한다. 고정 통계를 저장했다 재사용하는 경우에도 `floor_stats`를 공개해 같은 방어를 받게 한다.
3. **표본 부족은 NaN** — 코호트 안 유효값이 `min_cohort_n`(5) 미만인 피쳐는 통계를 내지 않는다. n=2로 낸 MAD로 z를 만드는 것보다 정직하고, `n_z_valid`에 반영되어 하류에서 추적된다.
4. **z 상한** (`z_cap` = 30) — 최종 방어선. **포화값이므로 z 크기를 심각도로 읽으면 안 된다.**

**집계 (`aggregate_z`)** — 피쳐가 수십 개면 하나쯤은 늘 튄다 (다중비교). 두 정의를 **둘 다** 계산해서 컬럼으로 남긴다. 어느 것을 판정에 쓸지는 정하지 않는다.

| 컬럼 | 정의 | 성질 |
|---|---|---|
| `z_max` | 최대 directed z (= `top_feature`의 top_z) | 하나만 튀어도 걸린다 |
| `z_top{k}_kth` | **k번째로 큰** z | 임계 초과 = "k개가 동시에 넘었다". 다중비교 방어가 온전한 대신 **반응 피쳐가 k개 미만인 실패를 놓친다** |
| `z_top{k}_mean` | 상위 k개의 평균 | 놓치는 실패는 적지만, 하나가 상한까지 포화하면 평균이 여전히 커서 방어가 온전하지 않다 |

대가는 실측된다: CD 하나를 자기 중점 기준으로 회전시키면 `obliquity`와 `angle_resid_seq` **둘만** 반응해 `z_top3_kth`에서 사라지고 `z_top2_kth`에서만 잡힌다 (§7.4).

- **NaN 주의**: `np.sort`는 NaN을 맨 뒤(최댓값 자리)로 보낸다. −inf로 치환하지 않으면 상위 k개가 전부 NaN이 되어 결과가 NaN — CD 1개짜리 카테고리(관계 피쳐 전부 NaN)에서 실제로 났던 버그. 유효 피쳐 수를 `n_z_valid`로 함께 기록한다.

**물리 단위 절대 임계 (`abs_flags`)** — 값이 0에 몰려 MAD가 구조적으로 죽는 피쳐(`bdist_s/e`, `obliquity`)와 원형 잔차(`angle_resid`)는 물리 단위 절대 임계가 해석도 쉽고 안정적이다. 그런 피쳐도 **z는 z대로 계산해서 기록**하고, 어느 쪽을 쓸지는 판정 단계에서 정한다. `abs_flags(df, thresholds)`는 **임계값에 기본값을 두지 않는다** — 공정 스펙 근거가 있는 값은 사용자만 안다. 비교 방향은 registry의 `worse_when`을 따른다.

**최빈값 불일치 (`mode_flags`)** — `n_cd`처럼 **본질적으로 이산이고 규칙적인** 값에는 z가 아예 맞지 않는다. 카테고리 안에서 늘 같은 값이면 MAD가 0이라 분모가 `mad_floors`가 되고, 그러면 "CD 1개 누락 = z 2.0"처럼 **하한이 그대로 답이 된다** — z가 정보를 담고 있지 않다는 뜻이다. 그룹 최빈값과 다르면 플래그하는 편이 임계도 필요 없고 해석도 명확하다.

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
- 참고 구성: `alarm = (z_top3 > 임계) OR (n_abs_flags > 0)`. **cdqc도 사용자 스크립트도 z와 플래그를 계산해서 내보낼 뿐 판정하지 않는다.** 임계값을 라이브러리에 박지 않는다.

클러스터링은 판정이 아니라 실패 모드 발견용 — 사용자가 플래그 CD의 z 벡터로 직접 (cdqc 범위 밖).

## 5. 파라미터 — `Params` dataclass

config 파일 없음. 필드: 샘플링(ribbon_half_w=2, win_frac=0.4, win_min/max_px=8/64, margin_px=6, grad_sigma_px=1.0, noise_patch_px=15, peak_suppress=6, npk_ratio=0.3, delta_sigma_sweep=(0.25…1.5)), 라벨 경계(label_inset_px=2.0), sanitize(min_cnr_for_ratio=1.0, min_rise_px=0.5), 시퀀스(local_window=9, method="robust_linear", min_seq_len=5), 통계(trim_frac=0.05, min_cohort_n=5, mad_floor_default=1e-6, mad_floors, log_features, direction_overrides, pooled_mad_frac=0.30, rel_scale_floor=0.01, abs_scale_floor=1e-3, z_cap=30). 기본값 근거는 §3–4.

`mad_floors`는 **이산/퇴화 피쳐의 MAD=0 방어**다 (단위는 피쳐 각자의 단위, log 피쳐는 로그 공간). 이산: `npk`/`n_cd`/`label_runs`/`max_run` 0.5. 물리 분해능: `delta_median` 0.05nm, `bdist_*` 0.05nm, `delta_B_*` 0.02px. **L2 요약값**(`cd_median`/`pitch_median`/`span_nm`/`traj_rms`/`angle_median`)에도 하한이 필요하다 — 레시피가 규칙적이면 카테고리 안에서 피치와 길이가 늘 같아 MAD가 구조적으로 0이 된다.

## 6. 공개 API

`import cdqc` 로 전부 접근. 시그니처와 사용 예제는 README.md가 기준.

- 피쳐: `extract_l1/l2/l3`, `mask_maps` + `boundary_features`(권장), `extract_mask_l3`, `extract_mask_image`
- 정규화: `cohort_stats`, `apply_z`, `robust_stats`, `floor_stats`, `cohort_z`(`return_stats=True`면 쓴 통계도 반환), `aggregate_z`, `agg_columns`, `abs_flags`, `mode_flags`, `threshold_from_quantile`
- 판정 연산: `top_feature`, `impact_nm`, `max_run`, `flag_rollup`, `hist_emd`
- 평가: `recall_at_fpr`, `localization_hit/rate`, `ablation_table` — 라벨은 사용자가 만들고 cdqc는 연산만
- 검증·진단: `inject_coords`, `injection_test`, `select_grad_sigma`, `trajectory_check`, `category_summary`
- 유틸: `to_nm`, `normalize_unit`, `infer_px_nm`, `ratio_cv`, `to_uint8`, `close_annotation`, `transform_coords`, `convention_scores`
- 메타: `FEATURES` registry, `Params`, `CdqcError`

## 7. 합성 데이터 생성기 + selftest (개발 전용)

### 7.1 이미지 모델

```
img = background(gradient) + Σ bands(erf edge profile, contrast, position)
      + fringe(옵션) + noise(Poisson-like) → clip → uint8
```

- 수직/약간 기울고 휜 밴드들. 밴드 경계가 엣지, 카테고리 = (좌, 우) 엣지 쌍.
- "레시피 출력" = 참 좌표 + 지터(`coord_jitter_px=0.3` — DL 컨투어 현실치. 이전 0.05는 테스트가 통과하도록 데이터를 맞춘 것이었다. 밴드 폭 산포도 `normal(56, 3)`로 넓혔다. **합성이 너무 쉬우면 PASS가 정보를 주지 않는다**). `value` = **참 길이 × px_nm** (주입 후에도 유지 → 좌표·값 불일치 경로 검증), 단위는 행마다 Å/nm 교대.
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
| `rotated_frame` | 시퀀스 전체 θ 회전 배치 + CD 수 감소 (회전 세그먼트도 엣지 위) — 총체적 실패 | L2 `angle_median`, `n_cd` | 개별 `delta_s`, `angle_resid`(시퀀스가 통째로 돌면 원형 잔차는 0) |
| `void_line` | 양 끝을 층 안쪽으로 당겨 계면을 안 건드리게 함 ("허공에 그은 선") | `bdist_ratio`, `bdist_s` | 같은 이미지의 `unaffected` CD |

### 7.3 selftest (`python -m cdqc.selftest`)

**공개 API만 사용해 조립** (사내 사용자 코드와 같은 모양 — API dogfooding). 베이스라인으로만 코호트 통계를 내고 주입 케이스를 그 기준으로 채점한다 (케이스 자체 코호트로 정규화하면 주입이 흡수됨).

- "반응해야 할" 칸: 강도 오름차순 단조 증가(슬랙 0.5) + 최종 z ≥ 3.
- "반응하면 안 되는" 칸: 전 강도 |z| < 1.5 — **계통 편향(부호 z) 기준**: both 피쳐의 directed z는 산포만 커져도 오르므로(예: defocus에서 delta 산포 증가) 부호 z 중앙값으로 잰다. 산포로 인한 개별 오플래그는 분위수 임계값이 관리.
- 전부 PASS면 exit 0. **재구조화·수정 후 이 58셀 PASS 재현이 회귀 기준.**
- selftest는 카테고리별 코호트를 `cohort_z`로 잡는다 (사내 사용자가 쓸 경로 그대로 — pooled 폴백과 z 상한을 포함해 dogfooding). 합성 마스크를 합쳐 라벨맵을 만들어 `mask_maps` + `boundary_features` 경로도 함께 검증한다.

### 7.4 주입 테스트 (`injection_test`) — 합성이 아니라 **실제 이미지**에도 쓴다

정상 좌표에 알려진 크기의 오류를 주입하고, **코호트 통계는 고정한 채** 주입된 CD의 z가 얼마나 오르는지 잰다 (주입된 시퀀스로 통계를 다시 내면 주입이 흡수된다). `extract(S, E) -> l3 DataFrame` 콜러블을 받으므로 사용자가 이미지·px_nm을 클로저로 잡으면 실데이터에 그대로 돌아간다 — 파일 I/O 없음 계약이 유지된다.

| kind | 내용 | 잡히는 경로 |
|---|---|---|
| `shift_one` | CD 하나의 S를 축 방향 이동 (px) | L3 잔차·delta 계열 → `z_top3_kth` |
| `rotate_one` | CD 하나를 **자기 중점** 기준 회전 (deg) | `obliquity`, `angle_resid_seq` **둘뿐** → `z_top2_kth`에서 멈춘다 |
| `rotate_frame` | **시퀀스 전체**를 `center` 기준 회전 (deg) | 모든 CD가 다른 자리로 → `delta`·`s_resid`·`bdist` 전부 |
| `swap` | 이웃 CD 둘의 E를 맞바꿈 (대응 오류) | L3 `obliquity` 포화 |
| `all_shift` | 시퀀스 전체 평행이동 (px) | **L2 `delta_median_s/e`** — 이웃 대비 잔차는 전부 0 |
| `drop_one` | CD 하나 삭제 | **L2 `n_cd`만** — L3는 기준선과 완전히 같다 |

**회전 두 종류는 서로 다른 실패다.** `rotate_one`은 끝점이 궤적 **진행 방향**으로 움직여서 법선 잔차(`s_resid`)가 원리적으로 못 본다. `rotate_frame`(기준 각도 오설정)은 모든 CD가 옮겨져 여러 피쳐가 함께 반응한다. 하나로 뭉뚱그리면 "각도 회전은 z_top3로 잡힌다/안 잡힌다"가 데이터마다 뒤집힌다.

반환 표에 `l3_top_feature` / `l2_top_feature`를 함께 낸다 — 집계 z만 보면 주입이 **의도한 경로로** 잡혔는지 알 수 없다. `*_rise`(기준선 대비 상승분)는 **같은 행 안에서** 계산된다: 케이스별 값을 따로 집계한 뒤 빼면 서로 다른 행의 중앙값을 빼게 되어 내부 모순이 생긴다.

주의할 점 둘:
- **z 크기를 심각도로 읽으면 안 된다.** 이동량이 커지면 오히려 탐색 창을 벗어나 delta가 원래 엣지를 놓쳐 z가 떨어진다. 탐지 여부만 판단할 것.
- `drop_one`을 z로 보면 민감도가 전적으로 `mad_floors["n_cd"]`(0.5)에 지배된다 — 1개 누락 = 정확히 2σ, 즉 **z가 정보를 담고 있지 않다**. `mode_flags`로 보는 것이 맞다 (§4.2b).

### 7.5 진단 지표 (`trajectory_check`, `category_summary`) — 계산해서 **기록만** 한다

카테고리별로 `linearity`(S 점들을 PCA 1축이 설명하는 비율), `monotonic`(측정 순서와 주축 위치의 순위 상관), 잔차 산포(`s_resid_mad_px` 등 — px 환산이라 배율 무관), 피쳐 중앙값(`cnr_med_*`, `rise_med_*`, `bdist_ratio_med`, `bgrad_med_*`, `label_runs_med`, `delta_B_med_*`, `delta_scatter_med_*`), `n_z_valid_med`, `z_top3_{kth,mean}_p99`를 표로 남긴다.

**궤적 진단은 트래젝토리 하나씩 계산해 중앙값을 낸다** (`traj_by`, 기본 `"image_id"`) — 여러 이미지의 좌표를 한 덩어리로 PCA에 넣으면 궤적이 겹쳐 `linearity`가 무의미해진다. `monotonic`은 진행 방향의 부호가 임의라 절댓값의 중앙값을 낸다.

관계 피쳐(`s_resid` 등)는 "측정 순서상 이웃한 CD는 공간적으로도 이웃"을 전제한다. 구조가 2D로 배열돼 있으면 그 전제가 깨지는데, **그래도 오경보는 나지 않는다** — 잔차가 전부 크면 카테고리 중앙값과 MAD가 같이 커져 z가 0 근처가 되기 때문이다. 대신 탐지력을 조용히 잃는다. `linearity`/`monotonic`은 어느 카테고리에서 그런지 **알아두기 위한** 지표이지 판정 기준이 아니다 — 한 세트에서 정한 0.98 같은 값을 코드에 박으면 그게 곧 부채다.

**이 값들로 피쳐를 빼거나 NaN 처리하는 자동 규칙을 만들지 않는다** (§11-2). 경고 출력은 해도 좋으나 계산 결과를 바꾸지 않는다.

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
  normalize.py           # cohort_z, floor_stats, aggregate_z, abs_flags
  validate.py            # inject_coords/injection_test, trajectory_check, category_summary
  features/
    l3.py                # 시퀀스 피쳐 조립 + sanitize (순수 함수)
    l1.py                # 이미지 피쳐
    mask.py              # 라벨 경계(mask_maps/boundary_features, 권장)
                         #  + 이진 마스크 정합(CD 레벨) + 마스크 형태(lm 레벨)
    registry.py          # 피쳐 이름/worse_when/사유코드/설명 — 단일 목록
  synth/                 # 개발 전용: generator.py(SynthParams, 메모리 내), inject.py
  selftest.py            # 개발 전용: python -m cdqc.selftest
scripts/                 # **사용자 코드 계층** (라이브러리 아님)
  run_batch.py           # 파일 로딩·좌표 변환·배치 루프·출력. 계산은 전부 cdqc
  feature_hist.py        # 분포 그림
tests/                   # pytest
pyproject.toml           # 의존성 4개 핀, pip install -e .
README.md                # 사용 예제 (API 기준 문서)
```

`scripts/`는 패키지에 포함되지 않는다(`pyproject`는 `cdqc*`만). **여기에 피쳐 계산이나 정규화 로직을 다시 쓰지 말 것** — 0.3까지 그랬고, 그래서 실데이터로 검증한 기능을 저장소 밖에서 잃을 뻔했다. 새 계산이 필요하면 cdqc에 넣고 여기서는 호출만 한다.

코딩 규칙: 피쳐 함수는 순수 함수 · 피쳐 추가 = registry 한 줄 + 계산 함수 하나 (registry가 정규화 방향·사유코드·selftest를 구동) · 예외는 E-코드 · 시드 고정 재현 · 프로파일 샘플링은 CD 배치로 `map_coordinates` 호출 최소화 · docstring에 피쳐의 의미 한 줄.

## 9. 왕복 프로토콜 (사외 개발 ↔ 사내 실행)

```
사외: 패치 + git push (selftest 58셀 PASS 상태로만)
  ↓ 사내: git pull (editable 설치라 재설치 불필요)
사내: 사용자 스크립트 실행 → 무차원 요약(z 분포, 플래그율, 상관, 에러 코드) 검토
  ↓ (규정상 반출 가능한 것만) 무차원 통계/서술/에러 코드
사외: 해석 → 다음 패치
```

- 사외로 나오는 것: 무차원 통계, PASS/FAIL, 카운트, E-코드, 서술. **뭘 내보낼지는 사용자가 규정 보고 판단. 애매하면 안 보낸다.**
- 사내 조정은 Params/임계값 등 사용자 코드에서 — cdqc 수정 없이 실험 가능해야 왕복이 준다.

## 10. 미결 사항 (알게 되면 이 문서 갱신)

**검증의 근본 한계 (먼저 인지할 것)**: 지금까지의 검증은 전부 정상 데이터 + 인위적 주입이다. "우리가 상상한 실패"는 잡지만 **"실제로 일어나는 실패"는 미검증**이다. 라벨된 오측정 데이터가 있어야 재현율을 말할 수 있다. 그리고 실데이터는 12장짜리 **한 세트뿐**이라 관찰된 수치 전부가 그 세트에 한정된다.

- ~~DL 마스크/확률맵 접근~~ → **해결 (변경 #03)**: segmentation PNG 확보, 마스크 피쳐 추가됨. 확률맵(soft mask)은 여전히 미확보 — 생기면 mgrad를 확률 가중으로 확장 검토.
- ~~`mask_grad_agree`의 판별력 부재~~ → **해결 (0.4)**: 분모를 국소 σ로, 이미지 프레임을 경계에서 제외.
- **−1px 계통 오프셋** — 거의 모든 카테고리에서 delta 중앙값이 약간 음수. 좌표 변환의 픽셀 중심 규약(`W/2` vs `(W−1)/2`)인지 DL 편향인지 미확인.
- **`delta_e` > `delta_s` 비대칭** — z_worst 기여가 245 대 87. 원인 미확인.
- **일부 카테고리의 `bdist_s` = 1.41nm** — 다른 정상 카테고리는 0.000.
- **`s_resid` MAD가 매우 작음** — 관찰된 세트에서 px_nm의 100분의 1 수준(0.000~0.086nm). 좌표가 고정 기준선 위에 배치된 것일 수 있으나 **다른 데이터로 확인 필요** — 이 관찰로 피쳐를 빼지 않는다 (§11-2).
- **중복 피쳐** — `delta_s`/`delta_e`/`delta_B_*`가 강상관이라 상위 k 집계를 자기들끼리 채울 수 있다. 판정 단계에서 다룰 문제.
- **카테고리별 임계** — 카테고리별 99% 분위가 4배까지 차이났다 (2.43~10.45).
- **절대 임계값의 근거** — 경험적으로 정한 값이고 공정 스펙 근거가 없다. 그래서 `abs_flags`는 기본값을 두지 않는다.
- **`obliquity`의 n=1 처리** — 현재 NaN이 아니라 0.0을 반환. NaN이 맞다.
- **`delta_B`/`delta_scatter`의 실데이터 수치는 S만 잰 것** — 원본 스크립트가 `delta_e`를 수집하지 않았다(설계가 아니라 누락). §7-2류 기여도표를 재계산하면 delta 계열 기여가 더 커질 것이다.
- **`bdist_s = 1.41nm` 카테고리** — 경계 표시가 전이 양쪽 픽셀이라 대각 이웃 아티팩트는 아니다. 실제 이탈인지 확인하려면 그 세트의 **px_nm 실측값**이 필요하다 (`read_px_nm()` 출력 한 줄).
- 카테고리별 공차(tolerance): 사용자가 스펙에서 가져와 impact와 비교. auto 없음.
- 레시피 최종 통계량 median/mean 여부: `impact_nm(stat=...)` 인자로 대응. 확인 필요.
- `local_window` 기본 9가 실데이터 시퀀스 길이 대비 적정한지: CD가 6개인 카테고리에서는 창이 시퀀스보다 크다. 결측률 보고 판단 (5~7 후보).
- 실데이터 라벨 확보 후 검증(재현율/국소화)은 사용자 스크립트로.

## 11. 설계 원칙 (판단이 필요할 때의 기준)

1. **카테고리 예외 목록을 만들지 않는다.** 카테고리 고유 성질은 카테고리별 z가 흡수한다 (§4.2b-1). 수정이 필요하면 **피쳐 자체의 일반 규칙**으로 고친다 — `sanitize`의 `min_cnr_for_ratio`/`min_rise_px`가 그 예다. 카테고리별 예외는 카테고리가 수십 개로 늘고 레시피가 바뀔 때마다 손봐야 하는 부채가 된다.
2. **전부 계산하고 전부 기록한다.** 한 데이터 세트의 관찰로 피쳐를 빼지 않는다. 다른 레시피에서는 좌표가 이미지에서 독립적으로 탐색될 수 있고, 그러면 같은 피쳐가 핵심 탐지기가 된다. 지금 단계는 피쳐 추출 파이프라인을 완성하는 것이지 판정을 확정하는 것이 아니다. z가 수학적으로 성립하지 않는 것(`angle` — 원형 변수)은 **변환**하고, 값이 0에 몰려 MAD가 죽는 것(`bdist_*`, `obliquity`)은 절대 임계를 **병행**하되 z도 같이 기록한다.
3. **사람이 정하는 파라미터를 최소화한다.** 임계값은 데이터의 분위수에서, σ는 스케일 합의에서 나온다. 파라미터 하나에 결과가 딸리면(delta ↔ grad_sigma_px) 스윕해서 없앤다.
4. **cdqc는 파일 I/O를 하지 않는다.** 순수 연산 라이브러리다. 이미지가 필요한 검증(`injection_test`)도 콜러블로 받는다.
5. **합성 통과는 버그 부재의 증거일 뿐 유효성의 증거가 아니다.** 합성을 만들 때는 일부러 어렵게 만든다.
6. **미해결은 미해결로 기록한다.** 나중에 "이거 왜 이렇지" 할 때 이미 알던 문제인지 새 문제인지 구분된다 (§10).
