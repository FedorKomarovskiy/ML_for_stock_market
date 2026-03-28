param(
    [ValidateSet("install", "run", "test", "build-notebook", "notebook")]
    [string]$Action = "run"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
Set-Location $RepoRoot

function Invoke-CheckedCommand {
    param([string[]]$CommandArgs)
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandArgs"
    }
}

switch ($Action) {
    "install" {
        & "C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe" -m venv (Join-Path $RepoRoot ".venv")
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r (Join-Path $RepoRoot "requirements.txt")
    }
    "run" {
        Invoke-CheckedCommand @("run_project.py", "--config", "config/btc_kucoin_hourly.json")
    }
    "test" {
        $env:PYTHONPATH = (Join-Path $RepoRoot "src")
        Invoke-CheckedCommand @("-m", "pytest", "tests", "-q")
    }
    "build-notebook" {
        Invoke-CheckedCommand @(".\scripts\build_notebook.py")
    }
    "notebook" {
        Invoke-CheckedCommand @("-m", "jupyter", "lab", "notebooks\btc_volatility_project.ipynb")
    }
}
