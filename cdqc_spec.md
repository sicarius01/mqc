# cdqc — TEM CD 측정 품질 판정 툴 스펙

버전 0.1 · 이 문서는 Claude Code에 넘기는 프로젝트 명세이자 합의된 설계의 단일 진실원(single source of truth)이다.
설계 논의가 바뀌면 이 문서를 먼저 고친다.

---

## 0. 한 문단 요약

DL 세그멘테이션 기반 레시피가 TEM 이미지에서 뽑은 CD 측정값 `(sx, sy, ex, ey)`가 제대로 된 것인지, 이미지와 측정 데이터에서 **의미를 아는 피쳐**를 뽑아 판정한다. 무지성 ML은 쓰지 않는다. 피쳐는 3단계 계층(CD → 카테고리 시퀀스 → 이미지)으로 뽑고, 코호트(레시피 × 카테고리) 내 robust 통계와 시퀀스 내 국소 잔차로 정규화한 뒤, 순차 게이트로 판정한다. 최종 판정 단위는 **이미지**, 국소화 단위는 **개별 CD**.

---

## 1. 운영 제약 (설계를 지배함)

| 제약 | 설계 결과 |
|---|---|
| 개발은 사외 PC(Claude Code), 실행은 사내 | 코드는 사내로 반입만 됨. 사외 PC는 실데이터를 절대 못 봄 |
| 사내 → 사외 반출 불가 | 코드가 **자가진단**해야 함. 관찰 결과의 대부분을 무차원 요약으로 출력 |
| 사내에서 pip install은 가능 (확인됨) | 설치는 pip로. 런타임 네트워크 호출은 여전히 0. 버전 핀, wheel 동봉은 백업 옵션 |
| DL 마스크/확률맵 없음, 좌표만 있음 | 마스크 ↔ 명암 정합은 `delta`로만 측정. 마스크 접근 전제 코드 경로 금지 |
| 개발 자체 검증은 합성 데이터로만 | 합성 생성기는 곁다리가 아니라 **핵심 자산**. 실패 모드 주입 가능해야 함 |
| GUI 없음 | 조정은 전부 `config.toml`. 결과는 정적 오프라인 HTML 리포트 |

**합성 테스트의 정직한 한계**: 합성에서 전부 PASS여도 "기계적으로 맞다"이지 "실제 TEM 실패를 잡는다"가 아니다. 합성은 버그를 잡고, 유효성은 실데이터 라벨로만 증명한다.

---

## 2. 데이터 계약

### 2.1 입력

**이미지**: 8-bit grayscale, `np.ndarray[uint8]`, shape `(H, W)`. `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`로 읽은 것.

**측정 레코드**: CD 하나 = 한 행. 최소 스키마:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `recipe_id` | str | 레시피 식별자. 사용자가 분리해서 넣음 |
| `image_id` | str | 이미지 식별자 (파일명 stem 등) |
| `image_path` | str | 이미지 파일 경로 (project root 기준 상대 또는 절대) |
| `category_id` | str | CD 카테고리 (A, B, C ...) |
| `cd_index` | int | ROI 내 측정 순서. **시퀀스 피쳐 전부가 여기 의존** |
| `sx, sy` | float | 엣지 S 좌표 (px, subpixel 가능) |
| `ex, ey` | float | 엣지 E 좌표 (px) |
| `px_nm` | float | 픽셀 크기 (nm/px). 이미지 단위로 같아도 행마다 반복 저장 |

- 파일 형식: CSV 또는 Parquet. 로더는 둘 다 지원.
- **CSV 읽기는 반드시 사용자 제공 함수 `read_nasca_csv(csv_path)` 경유** (사내 CSV에 보안 프로그램이 걸려 있음). 함수는 `cdqc/func.py`(**gitignore 대상, 사용자 소유** — pull 충돌 방지)에 두고 pandas DataFrame을 반환한다. 파일이 없으면 `cdqc/func_default.py`(일반 `pandas.read_csv`)로 폴백 — 사외 개발용. 사내 작성 템플릿은 `cdqc/func.py.example`. cdqc 내부에서 다른 경로로 CSV를 읽는 코드는 금지.
- 여러 레시피가 한 파일에 섞여 있어도 됨 (`recipe_id`로 분리).
- 컬럼명이 다르면 `[data.columns]`에서 매핑.

### 2.2 세그먼트의 의미 (확정)

`(sx,sy)`와 `(ex,ey)`는 **서로 다른 두 엣지 위의 점**이고, 둘을 잇는 선분 길이가 CD다. 선분은 엣지를 따라가는 게 아니라 **엣지 사이를 가로지른다**. 따라서:

- 세그먼트 축 방향 프로파일에 S에서 스텝, 중간 플래토, E에서 스텝이 있어야 정상.
- 같은 카테고리의 CD들은 `cd_index` 순으로 ROI를 슬라이딩하며 측정된 것. `{S_i}`와 `{E_i}`는 각각 **매끄러운 엣지 궤적(DL 컨투어의 샘플)**이어야 한다.

### 2.3 좌표 컨벤션 — 착수 시 반드시 확인

`(x, y)` = `(col, row)`를 기본으로 가정하되, `cdqc doctor`가 자동 탐지한다 (§6.1). 확인 전까지 어떤 이미지 피쳐도 신뢰하지 않는다.

---

## 3. 피쳐 설계

### 3.1 원칙

1. **추출기는 완전 범용.** 레시피가 뭔지 모른다. 모든 CD에 동일한 고정 길이 벡터.
2. **의미 부여는 코호트가 한다.** 정상/이상은 `(recipe_id, category_id)` 코호트 분포와 시퀀스 내 이웃이 결정.
3. **무차원화.** 세션 간 비교 가능하도록 노이즈·배경·이웃으로 나눈다.
4. **방향 메타데이터.** 피쳐마다 `worse_when = "low" | "high" | "both"`를 붙인다. 단측 z를 써야 할 것을 양측으로 쓰면 선명한 엣지가 이상으로 잡힌다.

### 3.2 L3 — 개별 CD 피쳐

프로파일 샘플링: 세그먼트 축을 따라 `S - margin_eff` ~ `E + margin_eff`를 subpixel 샘플링 (`scipy.ndimage.map_coordinates`, order=1). `margin_eff = max(margin_px, W/2 + 2)` — 바깥 여유가 탐색 창 절반보다 짧으면 창 크기만큼의 엣지 점프를 delta가 추적하지 못한다. 노이즈 억제를 위해 **국소 엣지 접선 방향**으로 `±ribbon_half_w` px 오프셋한 줄들을 평균(리본). 접선은 세그먼트 수직이 아니라 이웃으로 추정: `tangent_S(i) = normalize(S_{i+1} - S_{i-1})`. 창 크기 `W = clamp(win_frac * CD, win_min_px, win_max_px)`.

그래디언트 피크의 subpixel 정제는 포물선 보간이 아니라 **반치폭(≥ 0.5·peak) 구간 가중 센트로이드**: argmax 계열은 강도 의존 노이즈에서 밝은 쪽으로 계통 편향된다 (defocus로 피크가 넓고 낮을 때 두드러짐).

국소 노이즈 σ: 프로파일 주변 패치의 고주파 잔차 MAD × 1.4826. 전역 σ 쓰지 않음(TEM은 두께 편차로 전역이 무의미).

**이미지 증거 (엔드포인트 S, E 각각)**

| 피쳐 | 정의 | worse_when | 반응해야 하는 주입 실패 |
|---|---|---|---|
| `delta_s`, `delta_e` | 보고 좌표 ↔ 그래디언트 피크(포물선 subpixel) 부호 있는 거리, nm. **1순위** | both (코호트 중앙값 대비) | `edge_jump`, `systematic_bias`(중앙값으로) |
| `cnr_s`, `cnr_e` | 스텝 높이 / 국소 노이즈 σ | low | `low_contrast`, `noise_up` |
| `rise_s`, `rise_e` | 10–90% 상승폭, nm | high | `defocus` |
| `margin_s`, `margin_e` | 1등 피크 / 2등 피크 (1등 ±6 샘플 억제 후) | low | `double_edge` |
| `npk_s`, `npk_e` | `g > npk_ratio * peak`인 유의 피크 개수 | high | `double_edge` |
| `overshoot_s`, `overshoot_e` | 스텝 양쪽 오버슈트 진폭 / 스텝 높이 (프레넬 프린지 대리) | high | `defocus` (fringe 옵션) |
| `plateau_cv` | S–E 사이 플래토 변동계수 | high | `plateau_defect` |
| `pol_s`, `pol_e` | 스텝 극성 ±1 | 코호트 최빈값과 불일치 | `edge_jump`(옆 레이어로) |
| `edge_valid_s/e` | 피크가 창 경계에 걸리지 않음 (bool) | false | — |

**기하 / 시퀀스 잔차** (전부 **국소 추세** 대비. 전역 중앙값 쓰지 말 것 — 실제 테이퍼에서 전부 플래그됨)

| 피쳐 | 정의 | worse_when | 반응 |
|---|---|---|---|
| `cd_nm` | `hypot(ex-sx, ey-sy) * px_nm` | — (코호트) | `edge_jump`. oblique 과대(1/cosθ−1)는 코호트 CD 산포(공정 변동)에 묻힐 수 있음 — oblique 검출은 `cd_resid`/`obliquity` 몫 |
| `cd_resid` | 국소 robust 추세 대비 CD 잔차 (Hampel 또는 짧은 창 robust 회귀) | both | `edge_jump` |
| `s_resid`, `e_resid` | 각 궤적의 국소 적합 대비 법선 방향 잔차, nm | both | `edge_jump` (한쪽만) |
| `dstep_s`, `dstep_e` | 이웃 간 1차 차분의 **법선 성분** 크기, nm (유클리드 크기는 측정 피치를 재는 것 — 피치 변동이 전부 플래그됨. 법선은 점프에 강건한 접선(롤링 메디안)에서) | high | `edge_jump` |
| `obliquity` | 세그먼트 방향과 국소 엣지 접선이 이루는 각의 시퀀스 중앙값 대비 편차 (deg) | high | `oblique` |
| `curv_s`, `curv_e` | 궤적 국소 곡률 (3점). **기본 비활성** — 노이즈와 진짜 곡률 구분 불가, `s_resid`가 대체. 리포트용으로만 계산 | high | `edge_jump` |
| `angle` | 세그먼트 절대 각도 | — (코호트) | — |

### 3.3 L2 — 카테고리 시퀀스 피쳐 (image × category)

| 피쳐 | 정의 |
|---|---|
| `n_cd` | CD 개수. 코호트 최빈값 대비 |
| `frac_flagged` | L3에서 플래그된 비율 |
| `max_run` | 연속 플래그 최대 길이. **게이트가 아니라 사유 코드용** (뭉침=국소 이미지 손상, 산발=락온 실패) |
| `cd_median`, `cd_mad` | 코호트 대비 |
| `delta_median_s/e` | 시퀀스 전체 계통 편향. 개별 delta는 정상인데 전부 같은 방향이면 DL 편향 |
| `traj_rms_s/e` | 궤적 잔차 RMS |
| `impact_nm` | `abs(stat(전체) - stat(플래그 제외))`. `stat`은 config(`median`/`mean`/`trimmed_mean`) |

### 3.4 L1 — 이미지 피쳐 (측정 무관, OOD 탐지)

| 피쳐 | 정의 |
|---|---|
| `noise_sigma` | Laplacian 응답 MAD × 1.4826 |
| `sat_lo`, `sat_hi` | `mean(img==0)`, `mean(img==255)` |
| `dyn_range` | p99 − p1 |
| `hist_emd` | 코호트 중앙값 히스토그램과의 EMD (calibrate 시 저장) |
| `struct_energy` | 그래디언트 크기 p90 / noise_sigma |
| `tile_energy_cv` | 타일별 struct_energy의 CV (부분 손상) |
| `n_bad_categories` | 이미지 내 L2 게이트 발동 카테고리 수 (롤업) |

FFT 전역 샤프니스는 넣지 않는다 (TEM에서 구조 변화와 초점 변화가 분리 안 됨).

---

## 4. 정규화와 스코어링

### 4.1 두 층의 정규화 — 역할이 다름

| | 잡는 것 | 못 잡는 것 |
|---|---|---|
| **시퀀스 내 잔차** (같은 이미지·카테고리 이웃) | 국소 이상, 점프. 세션 변동에 완전 면역 | 시퀀스 전체가 균일하게 밀린 것 |
| **코호트 대비** (recipe × category, 데이터셋 전체) | 전체 편향, 이미지 단위 이상 | 국소 이상 (희석됨) |

둘 다 쓴다. 시퀀스 잔차 → 1차, 시퀀스 요약통계의 코호트 대비 → 2차.

### 4.2 코호트 통계

- 키: L3/L2는 `(recipe_id, category_id)`, L1은 `(recipe_id,)`
- `cd_index`는 키에 넣지 않고 공변량으로. 위치 의존 경향은 잔차 모델에서 제거
- robust z: `(x − median) / max(1.4826·MAD, mad_floor)`. `mad_floor`는 피쳐별 config
- 양수 heavy-tail 피쳐(`rise`, `margin`, `dstep`)는 log 후 z
- 오염 처리: 1차 z로 상위 `trim_frac`(기본 5%) 제거 후 재계산 1회
- 샘플 부족 폴백: `n < min_cohort_n`이면 `(recipe_id,)` → 전역. 폴백 여부를 `cohort_fallback_level` 컬럼에 기록, 폴백된 판정은 리포트에서 별도 표시

### 4.3 판정 구조 — 두 층, 다른 임계 철학

**CD 레벨 (느슨, 고재현율)**: 오경보 비용이 싸다(사람이 그 CD 하나 보면 끝). `t_soft` 임계. 출력 `(flag, reason_code, top_feature, z)`.
사유 코드 5개: `EVIDENCE_WEAK` / `POSITION_MISMATCH` / `SEQUENCE_JUMP` / `GEOMETRY_ODD` / `FOCUS_BLUR`.

**이미지 레벨 (엄격, 집계 증거)**: 재측정/재촬영 비용. 세 갈래 OR.

| 게이트 | 조건 | 사유 |
|---|---|---|
| **G0** | L1 이미지 품질 단측 z > `t_image` (어느 하나라도) | `IMAGE_BAD` — 이미지 자체 불량, 전체 무효 |
| **G1** | 어느 카테고리든 `impact_nm > tolerance_nm[category]` | `STAT_CONTAMINATED` — 보고 통계량 오염 |
| **G2** | 시퀀스 요약(`delta_median`, `cd_median`, `n_cd`)의 코호트 z > `t_seq` | `SYSTEMATIC_SHIFT` — 계통 실패. 개별 플래그 0개여도 잡힘 |

`impact_nm`은 **물리 단위**라 스펙 공차와 직접 비교 가능하다. CD 레벨 임계를 느슨하게 잡아도 무해한 CD 몇 개 더 걸리는 건 `impact`를 안 움직인다 — 이게 두 층을 분리하는 이유.

`max_run`으로 사유 세분화: `IMAGE_BAD_LOCAL`(뭉침) vs `LOCKON_SPORADIC`(산발).

### 4.4 클러스터링의 위치

판정이 아니라 **실패 모드 발견**용. 플래그된 CD만 모아 피쳐 공간에서 군집을 보고 사후에 사유 코드를 붙이는 보조 도구. `cdqc explore` 서브커맨드로 분리, 판정 경로에 넣지 않는다.

---

## 5. 설정 — `config.toml`

### 5.1 우선순위

```
CLI 플래그  >  config.toml 명시값  >  calibrated.toml  >  내장 기본값
```

- `config.toml`: 사용자가 손으로 편집. 저장소에 커밋.
- `calibrated.toml`: `cdqc calibrate`가 씀. **손으로 편집하지 않는다.** 코호트 통계와 자동 임계값이 들어감. gitignore.
- 임계값 항목에 문자열 `"auto"`를 쓰면 `calibrated.toml` 값을 따르고, 숫자를 쓰면 그 값으로 고정. 사용자가 손댄 것만 고정되고 나머지는 자동.
- 모든 출력에 `config` 해시와 `calibrated` 해시를 스탬프.

### 5.2 스키마 (전체)

```toml
[project]
root = "."                       # 모든 상대경로 기준. 사내에서 이것만 바꾸면 되게
name = "cdqc"
seed = 42                        # 합성/트림 등 모든 난수

[paths]                          # root 기준 상대 또는 절대
data_dir       = "data"          # 측정 레코드
image_dir      = "images"
output_dir     = "out"
cache_dir      = "out/cache"     # 피쳐 parquet (이미지 I/O는 딱 한 번)
calibrated     = "calibrated.toml"
labels         = "labels.csv"    # 선택. 검증용

[data]
format = "csv"                   # csv | parquet
image_ext = ".png"
[data.columns]                   # 사용자 컬럼명 → 표준명 매핑
recipe_id = "recipe_id"
image_id = "image_id"
image_path = "image_path"
category_id = "category_id"
cd_index = "cd_index"
sx = "sx"; sy = "sy"; ex = "ex"; ey = "ey"
px_nm = "px_nm"
[data.coords]
convention = "auto"              # auto | xy | rowcol ; auto면 doctor 결과 사용
origin = "zero"                  # zero | one
y_flip = false
scale = 1.0                      # 좌표가 다운샘플 기준이면 배수

[sampling]
ribbon_half_w = 2                # 접선 방향 ±px 평균
win_frac = 0.4                   # 창 = win_frac * CD
win_min_px = 8
win_max_px = 64
margin_px = 6                    # S/E 바깥 여유
grad_sigma_px = 1.0              # 그래디언트용 가우시안
noise_patch_px = 15              # 국소 노이즈 추정 패치
peak_suppress = 6                # 2등 피크 탐색 시 1등 주변 억제 폭 (샘플)
npk_ratio = 0.3                  # 유의 피크 기준 (peak 대비)

[sequence]
local_window = 9                 # 국소 추세 창 (홀수). 실제 구조 변화 스케일보다 짧게
method = "robust_linear"         # hampel | robust_linear. 러닝 메디안(hampel)은 기울거나
                                 # 휜 궤적에 계통 잔차를 남김 (베이스라인 z_s_resid p90:
                                 # hampel 9.8 vs robust_linear 1.6) — 기본은 국소 Theil-Sen
min_seq_len = 5                  # 이보다 짧으면 시퀀스 잔차 생략, 코호트만

[cohort]
key_l3 = ["recipe_id", "category_id"]
key_l1 = ["recipe_id"]
min_cohort_n = 200               # 미만이면 폴백
trim_frac = 0.05                 # 오염 제거 트림
mad_floor_default = 1e-6
[cohort.mad_floor]               # 피쳐별 오버라이드 (이산/퇴화 피쳐 MAD=0 방어)
npk_s = 0.5
npk_e = 0.5
n_cd = 0.5
sat_lo = 0.002                   # 픽셀 비율. 정상에서 전부 0 → MAD=0
sat_hi = 0.002
hist_emd = 0.5
frac_flagged = 0.05
overshoot_s = 0.02
overshoot_e = 0.02
max_run = 0.5
noise_sigma = 0.05               # gray level. uint8 양자화로 MAD=0 가능
dyn_range = 1.0
delta_median_s = 0.05            # nm. subpixel 분해능 아래 MAD 증폭 방지
delta_median_e = 0.05
[cohort.log_features]
names = ["rise_s","rise_e","margin_s","margin_e","dstep_s","dstep_e","cd_mad"]

[features]
enabled_l3 = "all"               # 또는 리스트. "all" = registry 기본 활성 전부
                                 # (curv_s/e는 기본 제외 — 명시 리스트로 켤 수 있음)
enabled_l2 = "all"
enabled_l1 = "all"
[features.direction]             # worse_when 오버라이드 (기본은 코드 내 메타데이터)
# cnr_s = "low"

[thresholds]
# 숫자 = 고정, "auto" = calibrated.toml
t_soft   = "auto"                # CD 레벨. auto 기준: 코호트 상위 soft_quantile
t_image  = "auto"                # G0
t_seq    = "auto"                # G2
[thresholds.auto]                # "auto"일 때 calibrate가 쓰는 정책
soft_quantile  = 0.90            # 코호트의 이 분위 이상이면 CD 플래그
image_quantile = 0.99
seq_quantile   = 0.99
[thresholds.tolerance_nm]        # G1. 카테고리별 공차. **필수 사용자 입력**, auto 없음
default = 1.0
# A = 0.8
# B = 1.5
[thresholds.impact]
stat = "median"                  # median | mean | trimmed_mean
trimmed_frac = 0.1

[gates]
enable_g0 = true
enable_g1 = true
enable_g2 = true
run_cluster_min = 3              # max_run ≥ 이면 IMAGE_BAD_LOCAL 사유

[report]
html = true
overlay_max_images = 50          # internal/overlay 개수 상한
overlay_flagged_only = true
summary_dir = "summary"          # 무차원 요약 (반출 검토 대상)
internal_dir = "internal"        # 이미지·nm·좌표 포함 (사내 전용)
[report.summary]                 # summary/에 뭘 쓸지. 기본은 무차원만
feature_stats = true
correlation = true
gate_counts = true
convention = true
selftest = true
version = true

[synthetic]                      # cdqc synth / selftest
n_images = 30
image_size = [512, 512]
n_categories = 3
cds_per_category = 20
px_nm = 0.5
base_contrast = 60
noise_sigma = 6.0
edge_rise_px = 1.5
coord_jitter_px = 0.3            # 정상 레시피 출력 지터 (DL 컨투어 현실치 0.3~1px)
fringe = false
[synthetic.inject]               # 실패 모드별 강도 스윕 (selftest 감도 곡선)
defocus         = [0, 1, 2, 4]           # rise 배수
low_contrast    = [1.0, 0.5, 0.25]       # 대비 배수
noise_up        = [1.0, 2.0, 4.0]
edge_jump_nm    = [0, 1, 2, 5]           # 최대값은 탐색 창(W/2) 안이어야 delta가 추적
systematic_bias_nm = [0, 1, 3]
oblique_deg     = [0, 10, 20]
missing_frac    = [0, 0.1, 0.3]
double_edge_offset_px = [0, 4, 8]
plateau_defect  = [false, true]
saturation      = [1.0, 2.0, 4.0]        # 대비 스트레치 배수 (클립 유발)
partial_damage  = [0, 0.25]              # 손상 타일 비율

[selftest]                       # selftest 판정 상수
n_images_per_case = 4            # (실패, 강도) 조합당 이미지 수
z_pass = 3.0                     # "반응해야 할" 피쳐의 최종 강도 z 하한
z_specificity = 1.5              # "반응하면 안 되는" 피쳐의 |z| 상한
mono_slack = 0.5                 # 단조 증가 판정 허용 슬랙 (z 단위)

[evaluate]
fpr = 0.05                       # 고정 이미지 오경보율 (재현율 측정 기준)

[logging]
level = "INFO"
file = "out/cdqc.log"
error_codes = true               # 예외에 E-XXX-NN 코드 부착
```

---

## 6. CLI

`cdqc <subcommand> [--config config.toml] [--root PATH] [--set key=value ...]`

`--set`은 TOML 경로를 점으로 (`--set thresholds.t_soft=3.0`). 여러 번 가능.

### 6.1 `cdqc doctor`

환경·스키마·좌표 점검. **실행 첫날 필수.**

- Python/패키지 버전, 네트워크 호출 없음 확인
- 스키마: 컬럼 존재, 타입, `cd_index` 중복/결측, `px_nm` 이미지 내 일관성
- 이미지: 존재, uint8, shape
- **좌표 컨벤션 자동 탐지**: 후보 8개 `{xy/rowcol} × {y_flip} × {origin 0/1}` (+ scale은 config에서)에 대해, **보고 좌표에서 세그먼트 축 방향 ±3px 안의 그래디언트 피크까지 거리(px, 포물선 subpixel)의 중앙값**을 점수로 (접선 방향 ±2px 리본 평균으로 노이즈 억제). 낮을수록 좋음: 맞으면 ~좌표 지터 수준, origin이 1px 어긋나면 ~1, 축이 뒤집히면 큼/OOB. ("평균 그래디언트 크기" 방식은 엣지 rise가 1px보다 넓으면 후보 간 차이가 원리적으로 작아 폐기.) 채택 조건: **1등 거리 ≤ 0.75px AND 2등/1등 비 ≥ 2** → `cache_dir/convention.json`에 저장. 비 < 1.3 또는 1등 > 1.5px → 경고 `W-CONV-01`. 점수표를 `out/summary/convention.txt`에 저장. (좌표 지터가 클수록 1등 거리가 올라가 origin 후보와의 비가 줄어든다 — 판별은 지터에 의해 근본적으로 제한됨.)
- 좌표 양자화 검사: 소수부 히스토그램. 정수/반픽셀에 몰려 있으면 `delta` 분해능 하한 경고
- 시퀀스: 카테고리별 `n_cd` 분포, `cd_index` 연속성

### 6.2 `cdqc synth`

합성 데이터셋 생성 → `data/synth/`. §7 참조.

### 6.3 `cdqc selftest`

`synth`로 만든 데이터에 전 파이프라인을 돌리고, **피쳐 × 주입 실패 감도 표**를 출력. 각 셀은 주입 강도에 따른 z-score 단조 증가 여부와 최종 강도에서의 z. 기대 표(§3.2의 "반응해야 하는 주입 실패" 컬럼)와 비교해 PASS/FAIL. 특이도도 본다: 반응하면 안 되는 조합에서 z가 튀면 FAIL.

출력: `out/summary/selftest.txt`. 전부 PASS면 exit 0.

### 6.4 `cdqc extract`

피쳐 추출 → `cache_dir/features_l3.parquet`, `_l2`, `_l1`. 이미지당 한 번만. 캐시 있으면 스킵(`--force`로 재추출). 코호트/임계값과 무관한 순수 함수.

### 6.5 `cdqc calibrate`

캐시된 피쳐로 코호트 통계 + auto 임계값 계산 → `calibrated.toml`. 히스토그램 템플릿, 카테고리별 `n_cd` 최빈값, 극성 최빈값도 여기 저장. `config.toml`은 건드리지 않는다.

**오염 한계**: 통계·auto 임계값은 "대부분 정상" 가정의 분위수라, 불량률이 높은 데이터로 캘리브레이션하면 임계값이 그대로 올라간다 (트림은 ~5% 오염까지만 방어). 그런 경우 `--baseline <파일>`(정상 image_id 목록, 한 줄에 하나, `#` 주석 허용)로 정상 이미지만 지정. 사용 여부는 calibrated.toml meta에 기록. 또한 `t_soft`는 피쳐 max의 분위수라 **피쳐 셋을 바꾸면 재캘리브레이션 필수**.

### 6.6 `cdqc run`

`extract`(캐시) → 정규화 → 판정 → 리포트. 출력:

```
out/
  cache/                     features_*.parquet
  internal/                  # 사내 전용
    per_cd.parquet           모든 L3 피쳐, z, flag, reason
    per_seq.parquet          L2
    per_image.parquet        L1 + 게이트 결과
    overlay/*.png            플래그 CD 오버레이 (config 상한)
    report.html              정적, 오프라인, self-contained
  summary/                   # 무차원 요약 — 반출 여부는 사용자가 규정 보고 판단
    feature_stats.txt        피쳐별 정규화 분포 (skew, kurtosis, MAD/median, 결측률)
    correlation.txt          피쳐 간 Spearman 상관 (|r|>0.9 쌍 강조)
    gate_counts.txt          게이트·사유 코드 분포, 플래그율
    convention.txt
    selftest.txt
    version.txt              커밋 해시, config 해시, calibrated 해시, 실행 시각
```

`--sweep thresholds.t_soft=2,2.5,3,4` 같은 스윕 지원. 한 왕복에 여러 설정 결과를 다 뽑는다.

### 6.7 `cdqc evaluate`

`labels.csv`(image_id, label, bad_cd_indices 선택)가 있으면:
- 탐지: 고정 이미지 오경보율(config, 기본 5%)에서 재현율, ROC 요약
- 국소화: 불량으로 맞힌 이미지 중 최고 z CD가 사람이 지목한 CD와 일치하는 비율
- 레이어 ablation: L3-geom만 / +image evidence / +L1 순으로 재현율 증분

### 6.8 `cdqc explore`

플래그된 CD의 피쳐 공간 클러스터링(HDBSCAN 또는 k-means, 선택 의존성). 실패 모드 발견용. 판정에 관여 안 함.

---

## 7. 합성 데이터 생성기

### 7.1 이미지 모델

```
img = background(gradient) + Σ layers(erf edge profile, contrast, position)
      + fringe(옵션: 엣지 양쪽 감쇠 사인)
      + noise(Poisson-like + Gaussian)
      → clip → uint8
```

- 층 구조: 수직 또는 약간 기울어진/휘어진 밴드 여러 개. 밴드 경계가 엣지.
- 각 카테고리 = 특정 두 엣지 쌍. `cds_per_category`개를 ROI를 따라 슬라이딩.
- Ground truth: 각 CD의 참 `(sx,sy,ex,ey)`.
- "레시피 출력" = 참 좌표 + 소량 노이즈 (정상). 실패는 주입으로.

### 7.2 주입 실패 (독립적으로 강도 조절 가능)

| 실패 | 구현 | 반응해야 할 피쳐 | 반응하면 안 되는 피쳐 (특이도) |
|---|---|---|---|
| `defocus` | 엣지 rise 배수 ↑ (+fringe 옵션) | `rise`, `overshoot` | `delta`, `cd_resid` |
| `low_contrast` | 층 대비 배수 ↓ | `cnr` | `rise`, `delta` |
| `noise_up` | 노이즈 σ 배수 ↑ | `cnr`, L1 `noise_sigma` | `delta_median` |
| `edge_jump` | i번째 CD의 S(또는 E)를 d nm 이동 | `delta_s`, `s_resid`, `dstep_s`, `cd_resid` | `e_*` |
| `systematic_bias` | 시퀀스 전체 S를 d nm 이동 | `delta_median_s` | 개별 `s_resid`, `dstep_s` |
| `oblique` | 세그먼트를 θ 회전 (양 끝 엣지 위 유지) | `obliquity`, `cd_resid` (국소 과대) | `delta`, `cnr` |
| `missing` | 일부 CD 삭제 | `n_cd` | 나머지 CD 피쳐 |
| `double_edge` | 엣지 근처에 약한 두 번째 엣지 추가 | `margin`, `npk` | `cnr` |
| `plateau_defect` | S–E 사이에 밝은/어두운 블롭 | `plateau_cv` | `delta` |
| `saturation` | 대비 과다 → 클립 | L1 `sat_hi/lo` | — |
| `partial_damage` | 이미지 일부 타일만 노이즈/블러 | L1 `tile_energy_cv`, L2 `max_run` | 다른 타일 CD |

### 7.3 selftest 기대 표

위 표가 곧 기대 표. `selftest`는 각 (실패, 강도) 조합에서 각 피쳐의 코호트 z를 재고, "반응해야 할" 칸은 단조 증가 + 최종 z ≥ 3, "반응하면 안 되는" 칸은 |z| < 1.5를 요구한다. 임계 상수는 `[selftest]` 섹션에 둔다.

캘리브레이션은 베이스라인(주입 없음)으로만 하고 주입 케이스를 그 기준으로 채점한다 — 케이스 자체 코호트로 정규화하면 주입이 흡수된다.

**특이도 판정은 계통 편향 기준**: `worse_when="both"` 피쳐의 directed z(=|z|)는 주입이 산포만 키워도 오른다 (예: defocus에서 delta 산포 증가). 특이도(silent) 칸은 부호 보존 z의 중앙값 |median z_signed| < 1.5로 잰다 — 편향 없는 산포 증가는 특이도 위반이 아니다. 산포로 인한 개별 CD 오플래그는 t_soft(분위 기반)가 관리한다.

---

## 8. 코드 구조

```
cdqc/
  __init__.py
  __main__.py            # cdqc CLI (argparse 또는 typer, 최소 의존)
  config.py              # TOML 로드, 우선순위 병합, "auto" 해석, 해시
  func.py                # 사용자 소유 read_nasca_csv (gitignore, 사내에서 작성)
  func.py.example        # 위 파일의 작성 템플릿 (커밋)
  func_default.py        # func.py 없을 때 폴백 (일반 read_csv, 사외용, 커밋)
  io.py                  # 레코드/이미지 로더, 컬럼 매핑, 좌표 변환
  errors.py              # 에러 코드 (E-CONF-xx, E-DATA-xx, E-CONV-xx, E-FEAT-xx)
  geometry.py            # 접선 추정, 국소 프레임, 잔차, 곡률, obliquity
  sampling.py            # ribbon 샘플러, 국소 노이즈
  evidence.py            # 프로파일 → delta/cnr/rise/margin/npk/overshoot/polarity
  features/
    l3.py                # CD 피쳐 (순수 함수, 코호트 무관)
    l2.py                # 시퀀스 집계, impact
    l1.py                # 이미지 피쳐
    registry.py          # 피쳐 이름, worse_when, log 여부, 설명 — 단일 목록
  normalize.py           # 시퀀스 잔차(Hampel/robust), 코호트 median/MAD, 폴백
  scoring.py             # 게이트, 사유 코드
  pipeline.py            # extract/run 오케스트레이션 (캐시 관리)
  calibrate.py
  synth/
    generator.py
    inject.py
  selftest.py
  report/
    html.py              # 정적 HTML (matplotlib PNG base64 인라인, 외부 리소스 0)
    summary.py           # summary/*.txt
    overlay.py
  doctor.py
  evaluate.py
  explore.py             # 선택 의존성
tests/                   # pytest. 합성 기반
config.toml              # 기본 설정 (커밋)
calibrated.toml          # gitignore
pyproject.toml           # 의존성 핀
wheels/                  # 선택. 사내 오프라인 설치용
README.md                # 사내 설치·실행 절차
```

### 8.1 의존성

필수: `numpy`, `scipy`, `polars`, `opencv-python-headless`, 표준 `tomllib`(py3.13 기준), `matplotlib`(리포트), `pandas`+`pyarrow`(`read_nasca_csv` 경유 CSV 읽기와 polars 변환).
선택: `hdbscan` 또는 `scikit-learn`(explore만).
런타임 네트워크 호출 금지. `pip download`로 wheels/ 채우는 스크립트 포함.

### 8.2 코딩 규칙

- 피쳐 함수는 순수 함수. 입력 (이미지, 레코드, config) → DataFrame. 전역 상태 없음.
- 피쳐 추가 = `registry.py`에 한 줄 + 계산 함수 하나. registry가 리포트·selftest·정규화 방향을 전부 구동.
- 예외는 반드시 `errors.py` 코드로. 사용자가 스택트레이스 대신 코드만 전달해도 진단 가능하게.
- 모든 출력 파일에 `version.txt` 동봉 (git hash, config hash, calibrated hash).
- 시드 고정. `synth`/`selftest` 재현 가능.
- 벡터화: 프로파일 샘플링은 CD들을 배치로 묶어 `map_coordinates` 호출 최소화. 병목이면 그 부분만 Rust(pyo3)로 — 초기엔 안 함.
- 타입 힌트, docstring에 피쳐의 **의미**를 한 줄로 (이게 "이해할 수 있는 피쳐"의 문서화).

---

## 9. 왕복 프로토콜 (사외 개발 ↔ 사내 실행)

한 라운드:

```
사외: 패치 + 버전 태그
  ↓ 반입
사내: cdqc doctor && cdqc selftest && cdqc run [--sweep ...]
      → out/summary/ 검토
  ↓ (사용자가 규정에 따라 반출 가능한 것만) 무차원 요약 / 서술 / 에러 코드
사외: 해석 → 다음 패치
```

- 한 라운드에 스윕까지 몰아서 돌린다. 실행은 자동이라 여러 개 돌리는 비용 ≈ 0, 왕복만 아낀다.
- 사외로 나오는 것: 무차원 통계, PASS/FAIL, 게이트 카운트, 에러 코드, 서술. **뭘 내보낼지는 사용자가 규정 보고 결정. 애매하면 안 보낸다.** "3번 게이트가 너무 많이 터짐" 수준의 서술만으로도 진행 가능하게 설계되어 있음.
- 사내에서 config 조정은 사용자가 직접. 코드 수정 없이 실험 가능해야 왕복이 줄어든다.

---

## 10. 착수 순서

| 단계 | 내용 | 산출물 | 비고 |
|---|---|---|---|
| 0 | 저장소 골격, config 로더, registry, 에러 코드, CLI 뼈대 | `cdqc doctor`가 합성 데이터에 돔 | |
| 1 | 합성 생성기 + 주입 + `synth` | `data/synth/` | **1순위**. 이게 있어야 이후 전부 자가검증 |
| 2 | L3 기하/시퀀스 피쳐 (좌표만, 이미지 안 읽음) + `impact_nm` + 게이트 G1 | `run`이 돔, `selftest` 기하 부분 PASS | 점프형 실패 대부분 |
| 3 | 샘플러 + `evidence.py` (`delta`,`cnr`,`rise`,`margin`,`npk`,`overshoot`,`polarity`) | `selftest` 증거 부분 PASS | **프로젝트의 승부처** |
| 4 | L1 이미지 피쳐 + G0, G2, `calibrate`, "auto" 임계 | 전체 `selftest` PASS | |
| 5 | HTML 리포트, summary/, overlay, `--sweep` | | |
| 6 | `evaluate`, `explore` | | 사내 라벨 200~300장 확보 후 |
| 7 | 사내 첫 실행: `doctor` → 컨벤션 확정 → `extract` → `calibrate` → `run` | summary 검토 | 여기서부터 왕복 |

### 10.1 사내 첫날 체크리스트

1. `cdqc doctor` — 컨벤션 점수표 1등/2등 비 확인. 3배 미만이면 여기서 멈추고 상의.
2. 오버레이 20장 눈으로 확인 (사내에서만). 세그먼트가 실제 엣지 사이에 놓이는지.
3. 좌표 양자화 경고 확인.
4. `cdqc extract` → `cdqc calibrate` → `cdqc run`.
5. `summary/feature_stats.txt`에서 결측률·skew 확인. 결측률 높은 피쳐는 샘플러 창/마진 문제.
6. `summary/correlation.txt`에서 |r|>0.9 쌍 확인 → `[features.enabled_*]`에서 하나 제거 검토.
7. `summary/gate_counts.txt` 플래그율. CD 레벨 10~20%, 이미지 레벨 수 %가 초기 목표. 크게 벗어나면 `[thresholds.auto]` 분위 조정.

---

## 11. 미결 사항 (알게 되면 이 문서 갱신)

- DL 마스크/확률맵 접근: 담당자 회신 대기. 가능해지면 `evidence.py`에 마스크 경로 추가 검토 (현재는 없음 전제).
- 카테고리별 `tolerance_nm`: 사용자가 스펙에서 가져와야 함. auto 없음.
- 레시피 최종 통계량이 median인지 mean인지: `[thresholds.impact].stat`에 반영. 확인 필요.
- 이미지당 카테고리 수, 카테고리당 CD 수의 실제 분포: `doctor`가 첫 실행에서 알려줌.
