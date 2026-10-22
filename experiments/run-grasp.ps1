$ErrorActionPreference = 'Stop'
$python = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$evaluator = "$PSScriptRoot\..\scripts\grasp_evaluate.py"
$output = "$PSScriptRoot\grasp-evaluations"
if (-not (Test-Path -LiteralPath $PSScriptRoot -PathType Container)) {
    throw "Missing experiments directory: $PSScriptRoot"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

Write-Host "$(Get-Date -Format o) COORDINATOR pid=$PID"
foreach ($family in @('E', 'P')) {
    Write-Host "$(Get-Date -Format o) STAGE evaluate $family"
    & $python -B -u $evaluator evaluate --family $family
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "$(Get-Date -Format o) STAGE cross"
& $python -B -u $evaluator cross --output "$output\results"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "$(Get-Date -Format o) FINISHED"
exit 0
