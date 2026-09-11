"""Selftest reports remain printable on legacy Windows console encodings."""
import io
import sys

import pytest

from cdqc import selftest


@pytest.mark.parametrize("encoding", ["cp949", "ascii", "utf-8"])
def test_report_preserves_console_encoding_and_result(monkeypatch, encoding):
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding=encoding, errors="strict", newline="\n")
    report = "총평: PASS 59, FAIL 0 — 결과 → PASS"
    monkeypatch.setattr(selftest, "run_selftest", lambda: (report, True))
    monkeypatch.setattr(sys, "stdout", stream)
    assert selftest.main() == 0
    stream.flush()
    assert stream.encoding == encoding
    assert output.getvalue().decode(encoding) == (report + "\n").encode(
        encoding, errors="backslashreplace").decode(encoding)


def test_report_supports_redirected_text_stream_and_failure_exit(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(selftest, "run_selftest", lambda: ("FAIL", False))
    assert selftest.main() == 1
    assert stream.getvalue() == "FAIL\n"
