<#
.SYNOPSIS
    Fuehrt die komplette ROBO_Benchmark-Pipeline lokal, Schritt fuer Schritt, aus
    (siehe README.md -> "Ablauf"). Bricht beim ersten Fehler ab.

.PARAMETER Instanzen
    Anzahl lokaler Instanz-Ordner (Standard: config.N_INSTANZEN).

.PARAMETER Workers
    Worker-Prozesse pro Instanz (Standard: config.WORKERS_PRO_INSTANZ).

.PARAMETER Funktion / Dimension
    Fuer den finalen Plot (plot_convergence.py) - welche Kombination geplottet wird.

.PARAMETER SkipPrepare
    Ueberspringt prepare_tasks.py + split_tasks.py (falls tasks.csv/instances/ schon vorhanden sind
    und nur weitergerechnet werden soll).

.EXAMPLE
    powershell -File run_pipeline.ps1
    powershell -File run_pipeline.ps1 -SkipPrepare
    powershell -File run_pipeline.ps1 -Funktion 1 -Dimension 2
#>

param(
    [int]$Instanzen = 0,        # 0 = config.N_INSTANZEN verwenden
    [int]$Workers = 0,          # 0 = config.WORKERS_PRO_INSTANZ verwenden
    [int]$Funktion = 1,
    [int]$Dimension = 2,
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$logPath = Join-Path $root "run_log.txt"
$python = "python"

function Invoke-Step {
    param([string]$Description, [string[]]$ScriptArgs)

    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    Write-Host "> $python $($ScriptArgs -join ' ')"

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "`n=== [$timestamp] $Description ==="
    Add-Content -Path $logPath -Value "> $python $($ScriptArgs -join ' ')"

    & $python @ScriptArgs 2>&1 | Tee-Object -FilePath $logPath -Append

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FEHLER in Schritt '$Description' (Exit-Code $LASTEXITCODE). Abbruch." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- Instanzen-/Worker-Anzahl aus config.py lesen, falls nicht per Parameter gesetzt ---
if ($Instanzen -eq 0) {
    $Instanzen = [int](& $python -c "import sys; sys.path.insert(0,'Pre_Processing'); import config; print(config.N_INSTANZEN)")
}
if ($Workers -eq 0) {
    $Workers = [int](& $python -c "import sys; sys.path.insert(0,'Pre_Processing'); import config; print(config.WORKERS_PRO_INSTANZ)")
}
Write-Host "Instanzen: $Instanzen | Workers/Instanz: $Workers"

# --- 1) Tasks erzeugen + aufteilen (einmalig zentral) ---
if (-not $SkipPrepare) {
    Invoke-Step "1a) Tasks erzeugen (prepare_tasks.py)" @("Pre_Processing/prepare_tasks.py")
    Invoke-Step "1b) Tasks aufteilen (split_tasks.py)" @("Pre_Processing/split_tasks.py", "--instanzen", "$Instanzen")
} else {
    Write-Host "SkipPrepare gesetzt - prepare_tasks.py/split_tasks.py werden uebersprungen." -ForegroundColor Yellow
}

# --- 2) Pro Instanz die Berechnung starten (lokal: nacheinander statt auf mehreren LRZ-Instanzen) ---
for ($i = 0; $i -lt $Instanzen; $i++) {
    $instanceDir = "data/instances/instance_$i"
    Invoke-Step "2) Berechnung instance_$i (run_instance.py)" @(
        "Main_Prozess/run_instance.py",
        "--instance-dir", $instanceDir,
        "--workers", "$Workers"
    )
}

# --- 2b) Fortschritt aller Instanzen aggregiert anzeigen ---
Invoke-Step "2b) Fortschritt pruefen (check_progress.py)" @(
    "Main_Prozess/check_progress.py",
    "--pattern", "data/instances/instance_*/results/status.json"
)

# --- 3) Ergebnisse zusammenfuehren und auswerten ---
Invoke-Step "3a) Ergebnisse zusammenfuehren (merge_results.py)" @(
    "Post_Processing/merge_results.py",
    "--pattern", "data/instances/instance_*/results/results.csv",
    "--out", "data/ergebnisse_gesamt.xlsx"
)

Invoke-Step "3b) Ins Wide-Format pivotieren (pivot_simple_regret.py)" @(
    "Post_Processing/pivot_simple_regret.py",
    "--pattern", "data/instances/instance_*/results/results.csv",
    "--out", "data/ergebnisse_wide.xlsx"
)

Invoke-Step "3c) Konfidenzintervall berechnen (compute_confidence_interval.py)" @(
    "Post_Processing/compute_confidence_interval.py",
    "--in", "data/ergebnisse_wide.xlsx",
    "--out", "data/ergebnisse_konfidenzintervall.xlsx"
)

Invoke-Step "3d) Konvergenzplot erzeugen (plot_convergence.py)" @(
    "Post_Processing/plot_convergence.py",
    "--in", "data/ergebnisse_konfidenzintervall.xlsx",
    "--funktion", "$Funktion",
    "--dimension", "$Dimension",
    "--out", "plots/f${Funktion}_d${Dimension}.png"
)

Write-Host ""
Write-Host "=== Pipeline komplett durchgelaufen. Log: $logPath ===" -ForegroundColor Green
