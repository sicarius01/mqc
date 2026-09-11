param(
    [string]$Python = "",
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) { $Python = $venvPython }
    else { $Python = "python" }
}
& $Python -c "import cdqc, streamlit, plotly, win32com.client, pythoncom, ncempy"
if ($LASTEXITCODE -ne 0) {
    Write-Host "GUI dependencies are missing. Run:" -ForegroundColor Yellow
    Write-Host "  $Python -m pip install -r requirements-gui.txt"
    Write-Host "Offline installation: see GUI_GUIDE.md"
    exit 1
}
$headless = if ($NoBrowser) { "true" } else { "false" }
Write-Host "CDQC Workbench: http://127.0.0.1:$Port"
Write-Host "Keep this window open. Press Ctrl+C to stop."
& $Python -m streamlit run (Join-Path $PSScriptRoot "cdqc_workbench\app.py") `
    --server.address 127.0.0.1 --server.port $Port --server.headless $headless `
    --browser.gatherUsageStats false
exit $LASTEXITCODE
