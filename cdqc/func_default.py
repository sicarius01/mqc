"""read_nasca_csv 기본 구현 — 사외 개발용 폴백.

cdqc/func.py(gitignore 대상, 사용자 소유)가 없을 때만 사용된다.
사내에서는 보안 프로그램을 통과하는 사용자 구현을 cdqc/func.py에 만든다
(cdqc/func.py.example 참조). 이 파일은 수정하지 말 것.

계약:
    read_nasca_csv(csv_path) -> pandas.DataFrame
"""

from __future__ import annotations

import pandas as pd


def read_nasca_csv(csv_path) -> pd.DataFrame:
    """CSV → pandas DataFrame. 일반 read_csv (보안 프로그램 없는 사외 환경용)."""
    return pd.read_csv(csv_path)
