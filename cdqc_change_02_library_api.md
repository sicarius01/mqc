# cdqc 변경 지시 #02 — 프레임워크 → 연산 라이브러리 재구조화 (확정)

상태: **확정 (2026-08-20).** §6 질문 전부 답변됨 — 이 문서 기준으로 재구조화한다.

---

## 0. 문제 인식

현재 구조는 CLI + config.toml + 출력 디렉토리 규약을 가진 **프레임워크**다.
cdqc가 CSV를 읽고, 워크플로우(doctor→extract→calibrate→run)를 소유하고,
calibrated.toml과 out/ 구조를 강제한다. 사용자는 이 틀 안에서만 움직여야 해서
사내에서 자기 코드에 끼워 넣을 수가 없다.

## 1. 목표 구조 (요구사항 재진술)

| | 사용자 (사내) | cdqc (라이브러리) |
|---|---|---|
| CSV/이미지 읽기 | O (자기 함수) | **안 함** |
| 데이터 정리·컬럼 매핑·단위 변환 | O | 안 함 (유틸 함수만 제공) |
| **피쳐 계산** (CD/시퀀스/이미지) | 호출만 | **O — 핵심** |
| **robust 통계·z 정규화** | 호출만 | **O — 핵심** |
| 코호트 정의 (어떤 행이 한 코호트인지) | O | 안 함 |
| 임계값·공차·판정 기준 | O | 안 함 (연산 헬퍼만) |
| 결과 저장·리포트·운영 | O | 안 함 |

cdqc의 모든 공개 함수는 **순수 함수**: 정해진 포맷의 인자 → 연산 → 리턴.
파일 I/O 없음, config 파일 없음, 전역 상태 없음, 네트워크 없음.

## 2. 사용 시나리오 (사내에서 사용자가 짜는 코드의 모양)

```python
import cv2
import pandas as pd
import cdqc

p = cdqc.Params()                    # 연산 파라미터 (dataclass, 기본값 내장)

# ── 1) 데이터 준비: 전부 사용자 코드 ─────────────────────────────
df = my_load_and_clean()             # 사내 CSV → DataFrame, 컬럼 정리도 사용자가
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# ── 2) 피쳐 계산: cdqc 호출 ──────────────────────────────────────
rows = []
for (iid, cat), g in df.groupby(["image_id", "category_id"]):
    f = cdqc.extract_l3(                     # 시퀀스(이미지×카테고리) 하나 → CD별 피쳐
        img=images[iid],                     # np.ndarray(uint8) 또는 None(기하만)
        S=g[["sx", "sy"]].to_numpy(),        # (n,2), x=col/y=row, 측정 순서대로 정렬
        E=g[["ex", "ey"]].to_numpy(),
        px_nm=0.5,
        value_nm=g["value_nm"].to_numpy(),   # 선택 (value_mismatch용)
        params=p)
    rows.append(f.assign(image_id=iid, category_id=cat))
l3 = pd.concat(rows)

l2 = cdqc.extract_l2(l3, group_cols=["image_id", "category_id"], params=p)
l1 = pd.DataFrame([{"image_id": iid, **cdqc.extract_l1(im, params=p)}
                   for iid, im in images.items()])

# ── 3) 정규화: 코호트는 사용자가 잘라서 넣음 ─────────────────────
stats = cdqc.cohort_stats(l3_normal, params=p)      # 정상 데이터로 통계
z = cdqc.apply_z(l3, stats, params=p)               # directed z (+부호 z) 컬럼 추가

# ── 4) 판정·운영: 전부 사용자 코드. cdqc는 연산 헬퍼만 ───────────
flag = z["z_top"] > my_t_soft                        # 기준은 사용자가 정함
impact = cdqc.impact_nm(g_cd_nm, flag_arr, stat="median")
run = cdqc.max_run(flag_arr)
```

## 3. 공개 API (제안)

### 3.1 피쳐 계산 (핵심)

| 함수 | 입력 | 출력 |
|---|---|---|
| `extract_l3(img, S, E, px_nm, value_nm=None, params)` | 한 시퀀스. img는 uint8 2D 또는 None | CD별 피쳐 DataFrame (delta/cnr/rise/margin/npk/overshoot/plateau/pol/edge_valid + cd_nm/잔차/dstep/obliquity/…) |
| `extract_l2(l3_df, group_cols, params)` | L3 결과 (여러 시퀀스 가능) | 시퀀스별 요약 (n_cd, cd_median/mad, delta_median, traj_rms) |
| `extract_l1(img, params)` | 이미지 하나 | dict (noise_sigma, sat, dyn_range, struct_energy, tile_energy_cv, hist) |

### 3.2 통계·정규화 (핵심)

| 함수 | 역할 |
|---|---|
| `robust_stats(x, trim_frac)` | 트림 1회 적용 (median, MAD) |
| `cohort_stats(df, feature_cols=None, params)` | 피쳐별 robust 통계 (+극성 최빈값). **코호트 분리는 호출 전에 사용자가** — 함수는 받은 행 전체를 한 코호트로 취급 |
| `apply_z(df, stats, params)` | directed z (worse_when 반영, log 피쳐 변환, mad_floor) + 부호 z 컬럼 추가 |

### 3.3 판정 연산 헬퍼 (기준은 전부 인자 — cdqc가 정하지 않음)

| 함수 | 역할 |
|---|---|
| `impact_nm(cd_nm, flags, stat="median", trimmed_frac=0.1)` | 플래그 제외 시 통계량 변화 (nm) |
| `max_run(flags)` | 연속 플래그 최대 길이 |
| `hist_emd(hist, template)` | 히스토그램 EMD |
| `top_feature(z_df, feature_cols=None)` | 행별 최대 directed z와 그 피쳐/사유 코드 |

### 3.4 메타데이터·유틸 (선택 사용)

| 항목 | 역할 |
|---|---|
| `FEATURES` | 피쳐 registry 공개: 이름, worse_when, 사유 코드, log 여부, 설명 |
| `Params` | 모든 연산 파라미터 dataclass (샘플링 창/리본/국소 추세 창 등. config.toml 대체) |
| `detect_convention(imgs, coords_list)` | 좌표 컨벤션 8후보 점수표 반환 (판단은 사용자) |
| `normalize_unit(s)` / `to_nm(value, unit)` | Å/nm 인식·변환 |
| `infer_px_nm(S, E, value_nm)` | value/기하 비 중앙값 역산 (이미지 하나 분량) |

### 3.5 입력 포맷 계약 (정해진 포맷)

- 좌표: `(n,2)` float ndarray, **(x=col, y=row), zero-origin, 이미지 픽셀 단위**.
  다른 컨벤션이면 넣기 전에 사용자가 변환 (detect_convention은 참고용).
- 시퀀스: **측정 순서대로 정렬해서** 넣는다 (시퀀스 피쳐가 순서에 의존).
- 이미지: `np.ndarray[uint8]`, shape (H, W).
- 길이 단위: 함수 경계에서 항상 **nm** (변환은 유틸로 지원).
- DataFrame은 **pandas** 기준 입출력 (read_nasca_csv가 pandas 반환이므로).
  내부 구현이 polars를 쓰더라도 경계에서는 pandas.

## 4. 기존 자산 처분

| 기존 | 처분 |
|---|---|
| geometry/sampling/evidence/features/normalize (연산 코어) | **유지** — API 밑에서 그대로 재사용. 검증된 수치 로직 손대지 않음 |
| io.py (CSV 로더, 컬럼 매핑, px_nm 역산, cd_index 생성) | **삭제** (좌표 변환·단위·역산은 §3.4 유틸로 축소) |
| func.py / read_nasca_csv 계약 | **삭제** — CSV 읽기는 애초에 cdqc 밖 |
| config.toml / calibrated.toml / Config | **삭제** — Params dataclass로 대체. 저장·로드는 사용자 자유 |
| CLI(__main__), doctor, calibrate, pipeline, report/, evaluate, explore | **사용자 경로에서 제거.** 판정 로직 중 재사용 가치 있는 것(§3.3)만 함수로 남김 |
| synth/, selftest, tests/ | **유지 (개발 내부용)** — 사외에서 연산의 감도/특이도를 검증하는 용도. 새 API를 직접 호출하도록 재작성. 사내 사용과 무관 |
| 에러 코드 체계 | 축소 유지 — 입력 검증 실패 시 코드 붙은 예외 (E-ARG-xx) |

## 5. 이행 순서

1. `Params` dataclass + 공개 API 함수 시그니처 확정 (§3)
2. 기존 연산 코어를 API 뒤로 재배선 (수치 로직 무변경)
3. selftest/tests를 새 API 직접 호출로 재작성 → **감도/특이도 39셀 PASS 동일 재현이 이행 완료 기준**
4. 프레임워크 부품 제거
5. README를 "사내 사용 예제 코드" 중심으로 재작성, cdqc_spec.md 갱신

---

## 6. 질문 → 답변 (확정)

1. **입출력 타입**: pandas DataFrame 확정.
2. **호출 단위**: **CD 카테고리(시퀀스) 단위** — 배치 편의 함수 없음, 루프는 사용자가.
3. **판정 헬퍼**: impact/max_run/top_feature 제공 (참고용으로 사용).
4. **분위수 임계값 유틸**: `threshold_from_quantile` 제공.
5. **배포 형태**: 사내에서 클론 후 `pip install -e .` (editable) — pull 받으면 재설치 없이 반영.
   `cdqc/` 폴더 복사 + 스크립트 옆 배치도 동작은 함.

## 7. 부수 결정 (재구조화하면서 확정)

- **의존성 축소**: polars/pyarrow/matplotlib 제거 — numpy, scipy, pandas,
  opencv-python-headless 4개만 남는다 (리포트·프레임워크가 없어졌으므로).
- synth/selftest는 파일 I/O 없이 **메모리 내**로 동작 (개발 검증 전용,
  `python -m cdqc.selftest`로 실행). PNG/CSV 저장은 눈 확인용 옵션.
- CLI 엔트리포인트(pyproject [project.scripts]) 제거.
