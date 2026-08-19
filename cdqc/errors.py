"""에러/경고 코드 체계.

사내 실행 시 사용자가 스택트레이스 대신 코드만 전달해도 사외에서 진단 가능해야
한다 (spec §8.2). 모든 예외는 CdqcError(code=...)로 던지고, 경고는 warn()으로
로그에 W-코드를 남긴다.
"""

from __future__ import annotations

import logging

log = logging.getLogger("cdqc")

# 코드 → 설명. 리포트/문서화의 단일 목록.
ERROR_CODES: dict[str, str] = {
    # 설정
    "E-CONF-01": "config.toml 파일을 찾을 수 없음",
    "E-CONF-02": "config 키가 스키마에 없음 (오타 가능성)",
    "E-CONF-03": "config 값 타입/범위 오류",
    "E-CONF-04": "--set 구문 오류 (key=value 형식이어야 함)",
    "E-CONF-05": "임계값이 'auto'인데 calibrated.toml이 없음",
    # 데이터
    "E-DATA-01": "측정 레코드 파일을 찾을 수 없음",
    "E-DATA-02": "필수 컬럼 누락",
    "E-DATA-03": "컬럼 타입 변환 실패",
    "E-DATA-04": "이미지 파일을 찾을 수 없음",
    "E-DATA-05": "이미지가 8-bit grayscale이 아님",
    "E-DATA-06": "cd_index 중복 (image, category 내)",
    "E-DATA-07": "px_nm이 이미지 내에서 일관되지 않음",
    "E-DATA-08": "레코드가 0건",
    # 좌표 컨벤션
    "E-CONV-01": "좌표가 이미지 범위를 벗어남 (컨벤션 오류 가능성)",
    "E-CONV-02": "convention='auto'인데 doctor 탐지 결과가 없음",
    # 피쳐
    "E-FEAT-01": "알 수 없는 피쳐 이름 (registry에 없음)",
    "E-FEAT-02": "피쳐 추출 실패",
    "E-FEAT-03": "피쳐 캐시가 없음 (cdqc extract 먼저 실행)",
    # 캘리브레이션
    "E-CAL-01": "calibrated.toml 파싱 실패",
    "E-CAL-02": "캘리브레이션에 필요한 피쳐 캐시가 없음",
    # 합성
    "E-SYN-01": "합성 설정 오류",
    # 평가
    "E-EVAL-01": "labels 파일을 찾을 수 없음",
    "E-EVAL-02": "labels 스키마 오류",
    # explore
    "E-EXPL-01": "explore 선택 의존성(scikit-learn) 없음",
    # 경고
    "W-CONV-01": "좌표 컨벤션 점수 1등/2등 비가 낮음 — 수동 확인 필요",
    "W-CONV-02": "좌표 양자화 감지 — delta 분해능 하한 있음",
    "W-COH-01": "코호트 샘플 부족 — 상위 레벨로 폴백",
    "W-CAL-01": "calibrated.toml 없음 — 현재 데이터로 즉석 캘리브레이션 (자기참조 주의)",
    "W-SEQ-01": "시퀀스가 min_seq_len 미만 — 시퀀스 잔차 생략",
    "W-FEAT-01": "피쳐 결측률 높음",
}


class CdqcError(Exception):
    """코드가 부착된 cdqc 예외. str()에 코드가 항상 포함된다."""

    def __init__(self, code: str, detail: str = ""):
        if code not in ERROR_CODES:
            raise ValueError(f"unregistered error code: {code}")
        self.code = code
        self.detail = detail
        msg = f"[{code}] {ERROR_CODES[code]}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


def warn(code: str, detail: str = "") -> None:
    """W-코드 경고를 로그에 남긴다."""
    if code not in ERROR_CODES:
        raise ValueError(f"unregistered warning code: {code}")
    msg = f"[{code}] {ERROR_CODES[code]}"
    if detail:
        msg += f" — {detail}"
    log.warning(msg)
