param(
    [Parameter(Mandatory)]
    [ValidateSet('numeric', 'grasp')]
    [string]$Experiment,

    [Parameter(Mandatory)]
    [ValidateSet('generate', 'evaluate', 'cross')]
    [string]$Stage,

    [ValidateSet('E', 'P')]
    [string]$Family,

    [int]$Candidates = 10,
    [string]$ArtifactsRoot,
    [string]$PlanForgeRoot,
    [string]$GraspRoot,
    [string]$WslDistribution
)

$ErrorActionPreference = 'Stop'
if (-not $ArtifactsRoot) { $ArtifactsRoot = Join-Path $PSScriptRoot '..\mulcia-tfm-artifacts' }
if (-not $PlanForgeRoot) { $PlanForgeRoot = Join-Path $PSScriptRoot '..\mulcia-tfm-PlanForge' }
if (-not $GraspRoot) { $GraspRoot = Join-Path $PSScriptRoot '..\mulcia-tfm-grasp' }
$ArtifactsRoot = (Resolve-Path $ArtifactsRoot).Path
$env:TFM_ARTIFACTS_DIR = $ArtifactsRoot
$env:TFM_GRASP_DIR = (Resolve-Path $GraspRoot).Path
$env:TFM_GRASP_SELECTION_DIR = Join-Path $PSScriptRoot 'outputs\grasp\selection'

if ($Stage -ne 'cross' -and -not $Family) {
    throw 'Generate and evaluate require -Family E or P.'
}
if ($Experiment -eq 'numeric' -and $Stage -eq 'cross') {
    throw 'The saved numeric cross-evaluation is published in artifacts; this launcher covers generation and family selection.'
}

if ($Experiment -eq 'numeric' -and $Stage -eq 'generate') {
    $parent = Join-Path $PSScriptRoot 'outputs\numeric-planning\selection'
    & uv run python -u scripts/gen_candidates.py $Family --split "$Family/prompt" --n $Candidates --output $parent
    exit $LASTEXITCODE
}

if ($Experiment -eq 'numeric' -and $Stage -eq 'evaluate') {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'WSL is required for numeric evaluation, but wsl.exe was not found.'
    }
    $selection = Join-Path $PSScriptRoot "outputs\numeric-planning\selection\$Family"
    $script = (Resolve-Path (Join-Path $PSScriptRoot 'scripts\numeric_evaluate.sh')).Path
    $planforge = (Resolve-Path $PlanForgeRoot).Path
    $wslOptions = @()
    if ($WslDistribution) { $wslOptions = @('--distribution', $WslDistribution) }
    $wslPaths = @($script, $selection, $selection, $ArtifactsRoot, $planforge) | ForEach-Object {
        $windowsPath = $_ -replace '\\', '/'
        $convertedOutput = & wsl.exe @wslOptions wslpath -a $windowsPath
        $conversionExit = $LASTEXITCODE
        $convertedPath = ($convertedOutput | Out-String).Trim()
        if ($conversionExit -ne 0 -or -not $convertedPath) {
            throw "WSL could not translate path: $_"
        }
        $convertedPath
    }
    & wsl.exe @wslOptions bash $wslPaths[0] $Family $wslPaths[1] $wslPaths[2] $wslPaths[3] $wslPaths[4]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv run python -m hllm_pipeline.rank (Join-Path $selection 'metrics.csv')
    exit $LASTEXITCODE
}

if ($Experiment -eq 'grasp' -and $Stage -eq 'generate') {
    $output = Join-Path $PSScriptRoot "outputs\grasp\selection\$Family"
    & uv run python -u scripts/grasp_gen_candidates.py --family $Family --n $Candidates --output $output
    exit $LASTEXITCODE
}

if ($Experiment -eq 'grasp' -and $Stage -eq 'evaluate') {
    $output = Join-Path $PSScriptRoot "outputs\grasp\selection\$Family"
    & uv run python -B -u scripts/grasp_evaluate.py evaluate --family $Family --output $output
    exit $LASTEXITCODE
}

$output = Join-Path $PSScriptRoot 'outputs\grasp\cross-evaluation'
& uv run python -B -u scripts/grasp_evaluate.py cross --output $output
exit $LASTEXITCODE
