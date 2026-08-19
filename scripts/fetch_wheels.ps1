# 사외에서 실행: 사내 오프라인 설치용 wheel을 wheels/에 내려받는다.
# 사내 파이썬과 동일한 버전(3.13, win_amd64) 기준.
param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$root = Split-Path -Parent $PSScriptRoot
& $Python -m pip download `
    -r (Join-Path $root "requirements.txt") `
    -d (Join-Path $root "wheels") `
    --only-binary=:all: `
    --python-version 3.13 `
    --platform win_amd64
if ($LASTEXITCODE -eq 0) {
    Write-Host "wheels/ 채움 완료. 반입 후: pip install --no-index --find-links wheels -r requirements.txt"
}
