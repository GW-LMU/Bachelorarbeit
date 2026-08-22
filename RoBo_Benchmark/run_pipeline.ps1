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
# .venv311 enthaelt cocoex + alle weiteren Pipeline-Abhaengigkeiten; das
# System-"python" auf dem PATH hat cocoex nicht installiert.
$python = Join-Path $root ".venv311\Scripts\python.exe"

# Zeitstempel dieses Laufs - wird am Ende benutzt, um data/ + plots/ in
# datierte Ordner (data/runs/<Zeitstempel>/, plots/runs/<Zeitstempel>/) zu
# verschieben, statt die Dateien direkt in data/ bzw. plots/ zu ueberschreiben.
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$dataDir = Join-Path $root "data"
$plotsDir = Join-Path $root "plots"
$runsDataDir = Join-Path $dataDir "runs"
$runsPlotsDir = Join-Path $plotsDir "runs"

function Archive-Run {
    param([string]$ArchiveId, [string]$Description)

    $items = Get-ChildItem -Path $dataDir -Exclude "runs" -ErrorAction SilentlyContinue
    $plotFiles = Get-ChildItem -Path $plotsDir -File -Filter "*.png" -ErrorAction SilentlyContinue
    $rootPlot = Join-Path $root "convergence_plot.png"
    $hasRootPlot = Test-Path $rootPlot

    if (-not $items -and -not $plotFiles -and -not $hasRootPlot) {
        return
    }

    Write-Host ""
    Write-Host "=== $Description -> data/runs/$ArchiveId, plots/runs/$ArchiveId ===" -ForegroundColor Yellow

    if ($items) {
        $dest = Join-Path $runsDataDir $ArchiveId
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($item in $items) {
            Move-Item -Path $item.FullName -Destination $dest -Force
        }
    }
    if ($plotFiles -or $hasRootPlot) {
        $dest = Join-Path $runsPlotsDir $ArchiveId
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($p in $plotFiles) {
            Move-Item -Path $p.FullName -Destination $dest -Force
        }
        if ($hasRootPlot) {
            Move-Item -Path $rootPlot -Destination $dest -Force
        }
    }
}

# --- 0) Vorherigen (noch nicht archivierten) Lauf zuerst wegsichern -------
# Zeitstempel dafuer: letzte Aenderungszeit der vorhandenen Dateien in data/,
# damit der Ordnername zum tatsaechlichen alten Lauf passt statt "jetzt".
# Nur wenn NICHT SkipPrepare: bei -SkipPrepare soll ja genau mit den
# vorhandenen tasks.csv/instances/ weitergerechnet werden, nicht wegarchiviert.
if (-not $SkipPrepare) {
    $prevItems = Get-ChildItem -Path $dataDir -Exclude "runs" -ErrorAction SilentlyContinue
    if ($prevItems) {
        $prevTimestamp = ($prevItems | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
        $prevRunId = Get-Date $prevTimestamp -Format "yyyyMMdd_HHmmss"
        Archive-Run -ArchiveId $prevRunId -Description "0) Vorherigen Lauf archivieren"
    }
}

function Invoke-Step {
    param([string]$Description, [string[]]$ScriptArgs)

    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    Write-Host "> $python $($ScriptArgs -join ' ')"

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "`n=== [$timestamp] $Description ==="
    Add-Content -Path $logPath -Value "> $python $($ScriptArgs -join ' ')"

    # Kein "2>&1" hier: bei nativen Programmen wie python.exe wickelt PowerShell 5.1
    # jede stderr-Zeile in einen ErrorRecord (NativeCommandError) und setzt $LASTEXITCODE/$?
    # faelschlich auf Fehler, auch wenn der Prozess mit Exit-Code 0 durchlaeuft. stdout wird
    # weiterhin geloggt, stderr laeuft unveraendert auf die Konsole durch.
    & $python @ScriptArgs | Tee-Object -FilePath $logPath -Append

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

Archive-Run -ArchiveId $runId -Description "4) Ergebnisse dieses Laufs archivieren"

Write-Host ""
Write-Host "=== Pipeline komplett durchgelaufen. Log: $logPath ===" -ForegroundColor Green
Write-Host "Ergebnisse dieses Laufs: data/runs/$runId" -ForegroundColor Green
Write-Host "Plots dieses Laufs: plots/runs/$runId" -ForegroundColor Green
