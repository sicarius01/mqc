"""에러 코드 체계 — 입력 검증 실패 시 코드 붙은 예외.

사내 사용자가 스택트레이스 대신 코드만 전달해도 사외에서 진단 가능하게
모든 예외는 CdqcError(code, detail)로 던진다.
"""

from __future__ import annotations

ERROR_CODES: dict[str, str] = {
    "E-ARG-01": "입력 배열 shape/길이 불일치",
    "E-ARG-02": "이미지가 8-bit grayscale 2D ndarray가 아님",
    "E-ARG-03": "알 수 없는 단위 값 — 추측하지 않고 중단",
    "E-ARG-04": "유효한 입력이 없음 (전부 NaN/길이 0 등)",
    "E-ARG-05": "알 수 없는 피쳐 이름 (registry에 없음)",
    "E-ARG-06": "파라미터 값 오류",
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
