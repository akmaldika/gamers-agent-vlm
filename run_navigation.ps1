param(
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptRoot
Set-Location $projectRoot

# Ordered navigation configs (without .yaml extension)
$configs = @(
    "01_navigation/01_5x5_empty_room",
    "01_navigation/01_15x15_empty_room",
    "01_navigation/01_30x30_empty_room",
    "01_navigation/01_single_turn_5",
    "01_navigation/01_single_turn_15",
    "01_navigation/01_single_turn_20",
    "01_navigation/01_straight_coridor_5",
    "01_navigation/01_straight_coridor_15",
    "01_navigation/01_straight_coridor_25",
    "01_navigation/01_t_junction_5",
    "01_navigation/01_t_junction_10",
    "01_navigation/01_t_junction_15"
)

Write-Host "Starting sequential execution of navigation configurations..." -ForegroundColor Green
Write-Host "Total navigation configs: $($configs.Count)" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop at any time" -ForegroundColor Yellow
Write-Host ""

for ($i = 0; $i -lt $configs.Count; $i++) {
    $config = $configs[$i]
    $current = $i + 1

    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host "Running navigation config $current/$($configs.Count): $config" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Cyan

    try {
        & $PythonExe "src/main.py" --config-name $config

        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: $config completed!" -ForegroundColor Green
        } else {
            Write-Host "ERROR: $config failed with exit code $LASTEXITCODE" -ForegroundColor Red
            $response = Read-Host "Continue with next configuration? (y/n)"
            if ($response -notin @('y','Y')) {
                Write-Host "Stopping execution." -ForegroundColor Red
                exit 1
            }
        }
    }
    catch {
        Write-Host "ERROR: $config threw an exception:`n$_" -ForegroundColor Red
        $response = Read-Host "Continue with next configuration? (y/n)"
        if ($response -notin @('y','Y')) {
            Write-Host "Stopping execution." -ForegroundColor Red
            exit 1
        }
    }

    Write-Host ""
    Start-Sleep -Seconds 2
}

Write-Host "All navigation configurations completed!" -ForegroundColor Green
