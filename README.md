# cdqc — TEM CD 측정 품질 판정 툴

DL 세그멘테이션 레시피가 뽑은 CD 측정값 `(sx, sy, ex, ey)`의 품질을,
의미를 아는 피쳐(3계층: CD → 시퀀스 → 이미지)와 순차 게이트로 판정한다.
설계 명세는 `cdqc_spec.md` (단일 진실원).

## 사내 설치

사내에서 `pip install`이 가능하므로 그대로 설치하면 된다 (런타임 네트워크
호출은 여전히 0 — 설치 때만 네트워크 사용).

```
python --version                       # 3.13.x 확인
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

> 백업 경로: 혹시 pip이 막힌 환경이면 사외에서 `scripts\fetch_wheels.ps1`로
> `wheels/`를 채워 반입한 뒤
> `pip install --no-index --find-links wheels -r requirements.txt`

## 사내 세팅 2가지 (사용자가 넣어야 하는 것)

### 1. `cdqc/func.py` — 보안 CSV 읽기 함수

사내 CSV는 보안 프로그램이 걸려 있어 일반 읽기가 안 되므로, 모든 CSV 읽기는
사용자 제공 함수 `read_nasca_csv()` 하나만 경유한다.

- **`cdqc/func.py`는 gitignore 대상** — 사내에서 작성해도 `git pull` 충돌이
  나지 않는다.
- 파일이 없으면 기본 구현(일반 `pandas.read_csv`)으로 폴백된다 (사외 개발용).
- 템플릿 복사 후 사내 로직을 채운다:

```
copy cdqc\func.py.example cdqc\func.py
```

```python
# cdqc/func.py — 예시 (자세한 변형은 cdqc/func.py.example 참조)
import pandas as pd

def read_nasca_csv(csv_path) -> pd.DataFrame:
    # 사내 보안 SDK가 평문 바이트를 돌려주는 경우:
    # import io
    # from nasca_sdk import decrypt_to_bytes
    # return pd.read_csv(io.BytesIO(decrypt_to_bytes(str(csv_path))))
    return pd.read_csv(csv_path)
```

계약: `read_nasca_csv(csv_path) -> pandas.DataFrame`. `csv_path`는 str 또는
`pathlib.Path`. 컬럼명 매핑/타입 캐스팅은 cdqc가 하므로 CSV 내용 그대로
반환하면 된다.

### 2. `config.toml` — 경로·컬럼 매핑·공차

- `[project].root` — 사내 데이터 위치 기준. 이것만 바꾸면 되게 설계됨
- `[data.columns]` — 사내 CSV 컬럼명 → 표준명 매핑
- `[thresholds.tolerance_nm]` — 카테고리별 공차 (**필수 사용자 입력**, auto 없음)

측정 레코드 CSV 예시 (CD 하나 = 한 행, `data/` 아래에 두면 전부 읽음):

```csv
recipe_id,image_id,image_path,category_id,cd_index,sx,sy,ex,ey,px_nm
RCP01,img_0001,images/img_0001.png,A,0,81.3,40.2,131.1,40.2,0.5
RCP01,img_0001,images/img_0001.png,A,1,81.4,63.0,131.0,63.1,0.5
RCP01,img_0001,images/img_0001.png,B,0,201.7,40.1,252.3,40.3,0.5
RCP01,img_0002,images/img_0002.png,A,0,80.9,40.0,130.8,40.1,0.5
```

- `(sx,sy)`·`(ex,ey)`는 서로 다른 두 엣지 위의 점, 잇는 선분 길이가 CD (px, subpixel 가능)
- `cd_index`: ROI 내 측정 순서 — 시퀀스 피쳐 전부가 여기 의존
- `px_nm`: 픽셀 크기 (nm/px), 이미지 안에서 동일해야 함
- 좌표가 (x,y)인지 (row,col)인지 몰라도 됨 — `cdqc doctor`가 자동 탐지
- 컬럼명이 다르면 `[data.columns]`에서 매핑 (파일을 고칠 필요 없음)

## 사내 첫날 체크리스트 (spec §10.1)

```
python -m cdqc doctor        # 좌표 컨벤션 점수표 — 2등/1등 비 3배 미만이면 멈추고 상의
python -m cdqc extract
python -m cdqc calibrate     # calibrated.toml 생성 (auto 임계값)
python -m cdqc run
# out/internal/overlay 20장 눈으로 확인 (세그먼트가 실제 엣지 사이에 놓이는지)
# out/summary/ 검토: feature_stats(결측률·skew), correlation(|r|>0.9),
#                    gate_counts(플래그율: CD 10~20%, 이미지 수 % 목표)
```

## 명령어

| 명령 | 역할 |
|---|---|
| `cdqc doctor` | 환경·스키마·좌표 컨벤션 점검 (실행 첫날 필수) |
| `cdqc synth` | 합성 데이터셋 생성 → `data/synth/` |
| `cdqc selftest` | 합성 전 파이프라인 감도/특이도 검증. 전부 PASS면 exit 0 |
| `cdqc extract` | 피쳐 추출 → `out/cache/*.parquet` (캐시, `--force` 재추출) |
| `cdqc calibrate` | 코호트 통계 + auto 임계값 → `calibrated.toml` |
| `cdqc run` | 정규화 → 판정 → 리포트 (`--sweep key=v1,v2,..` 지원) |
| `cdqc evaluate` | `labels.csv` 기반 재현율/국소화/ablation |
| `cdqc explore` | 플래그 CD 클러스터링 (선택 의존성 scikit-learn) |

공통 옵션: `--config config.toml` `--root PATH` `--set key=value` (여러 번 가능)

예: `python -m cdqc run --set thresholds.t_soft=3.0 --sweep thresholds.t_soft=2,2.5,3,4`

## 출력

- `out/internal/` — **사내 전용** (이미지·nm·좌표 포함): per_cd/per_seq/per_image
  parquet, `report.html`(오프라인 self-contained), `overlay/*.png`
- `out/summary/` — 무차원 요약 (반출 여부는 사용자가 규정 보고 판단):
  feature_stats, correlation, gate_counts, convention, selftest, version

## 개발 (사외)

```
python -m cdqc synth && python -m cdqc selftest   # 자가검증 (39개 감도/특이도 셀)
python -m pytest tests -q                          # 단위 테스트
```

합성 PASS는 "기계적으로 맞다"는 뜻이지 실제 TEM 실패를 잡는다는 증명이
아니다. 유효성은 사내 라벨로만 증명한다 (spec §1).
