# ROBO_NEU – Verteilte BO-Benchmark-Pipeline

Rechnet die BO-Benchmarks aus `ROBO/` auf 4 unabhaengigen LRZ-Instanzen (je
10 Kerne) parallel. Eine Task = eine (Parameterkombination × Tensorblock)
-Kombination; nur die Iterationsschleife innerhalb einer Task ist sequentiell,
alles andere laeuft parallel.

## Ablauf

**1) Einmalig zentral (z.B. auf deinem Rechner oder Instanz 0):**

```bash
python prepare_tasks.py          # erzeugt data/tasks.csv, data/tensor_noise.npy
python split_tasks.py --instanzen 4   # teilt in data/instances/instance_0..3 auf
```

**2) Pro Instanz: `data/instances/instance_<i>/` (enthaelt `tasks.csv` +
`tensor_noise.npy`) auf die jeweilige Instanz kopieren, z.B.:**

```bash
scp -r data/instances/instance_0 user@instanz0:~/robo_neu/
```

**3) Auf jeder Instanz die Berechnung starten:**

```bash
python run_instance.py --instance-dir ~/robo_neu/instance_0 --workers 10
```

Läuft der Prozess ab (Absturz, Neustart, SSH-Abbruch), einfach denselben
Befehl erneut ausfuehren – bereits erledigte Tasks werden anhand von
`results.csv` automatisch übersprungen.

## Fortschritt verfolgen

- Auf jeder Instanz läuft ein `tqdm`-Balken direkt in der Konsole.
- Jede Instanz schreibt laufend `results/status.json` (erledigt/gesamt/%,
  Tasks pro Sekunde, geschätzte Restzeit).
- Sobald die `status.json`-Dateien aller 4 Instanzen an einem Ort verfügbar
  sind (z.B. per `scp` zurückkopiert oder gemountet), aggregierter Überblick:

```bash
python check_progress.py --pattern "data/instances/instance_*/results/status.json"
```

Kann beliebig oft während der laufenden Berechnung ausgeführt werden.

## Ergebnisse zusammenführen

Nach Abschluss aller Instanzen (`results.csv` je Instanz zurückkopiert):

```bash
python merge_results.py --pattern "data/instances/instance_*/results/results.csv" --out data/results_gesamt.xlsx
python pivot_simple_regret.py --pattern "data/instances/instance_*/results/results.csv" --out data/df_gesamt_kombi_2.xlsx
python compute_confidence_interval.py --in data/df_gesamt_kombi_2.xlsx --out data/df_gesamt_ci.xlsx
```

`compute_confidence_interval.py` nimmt die Wide-Format-Datei von
`pivot_simple_regret.py` (eine Zeile je Kombination+Iteration, Tensorblöcke
nebeneinander in `Var_1..Var_n`) und ersetzt die `Var_n`-Spalten durch
`mean`, `std`, `ci_lower`, `ci_upper`, `ci_halfwidth` (t-Verteilung, Default
95 % Konfidenzniveau, per `--konfidenz` änderbar) — wieder im Wide-Format,
eine Zeile je Kombination+Iteration. Das ist die Grundlage für
Konvergenzgeschwindigkeits-Plots je Kombination: `mean` über `iteration`
als Linie, `ci_lower`/`ci_upper` als Konfidenzband.

## Konfiguration

Alle Parameter (Funktionen, Dimensionen, Modelle, Iterationen, Seeds, Anzahl
Instanzen/Worker) liegen zentral in `config.py`. Für den Zielumfang von
500.000+ Kombinationen dort `FUNKTIONEN`, `DIMENSIONEN`,
`SURROGATE_AKQUISITION_PAARE`, `N_SAMPLE_INIT` entsprechend erweitern.

## Implementierte Surrogatmodell-Varianten

Analog zu `waehle_und_berechne_variante()` in `ROBO/coco_benchmark.py` sind
in `core_bo.select_and_run_variant()` folgende Varianten angebunden (jeweils
angepasste Kopien der Skripte aus `ROBO/`, damit `ROBO_NEU` ohne Abhängigkeit
zum alten Ordner auf jede LRZ-Instanz kopiert werden kann):

| Surrogatmodell (`config.SURROGATE_AKQUISITION_PAARE`) | Akquisition | Skript | Einschränkung |
|---|---|---|---|
| `"GausianProzess"` | `EI` \| `UCB` \| `LCB` | `core_bo.py` (`run_gaussian_process`) | — |
| `"Imprecise_GausianProzess_Rodemann"` | `LCB` \| `GLCB` | `imprecise_gp.py` + `uncertainty_aware_bo.py` | nur `dim=1` |
| `"Relevance Pursuit"` | `EI` \| `UCB` \| `LCB` | `relevance_pursuit.py` | — |
| `"STABLEOPT"` | nur `UCB` | `stable_opt.py` | feste (ucb,lcb)-Konfidenzbänder intern |
| `"DRBO"` | nur `"distributionally robust UCB-Akquisition"` | `drbo.py` | letzte Spalte von `dim` = Kontextvariable (`context_dim=1` fest) |
| `"AIRBO"` | nur `UCB` | `airbo.py` | ⚠️ **quadratische Laufzeit, siehe Warnung unten** |

Um eine Variante mitzurechnen, in `config.py` die entsprechende Zeile in
`SURROGATE_AKQUISITION_PAARE` einkommentieren (bzw. eigene Zeile ergänzen).

### ⚠️ Bekannte Einschränkung: AIRBO-Laufzeit

`airbo.py` ist eine schlanke Eigenimplementierung nach Yang et al. 2023
(MMD-Kernel über simulierte Eingabeunsicherheits-Stichprobenwolken +
Nyström-Approximation + UCB), **nicht** der schwergewichtige GPyTorch-Code aus
`huawei-noah/HEBO` (eigene `ExactGP`-Unterklasse mit Latent-Mapping, GPU-Batch-
Kernel über mehrere Module – passt nicht in das einheitliche `run_*()`-Schema
dieses Projekts).

Gemessen wurden ca. 0,6 s/Iteration bei 6 Beobachtungen und bereits ca. 5 s/
Iteration bei 50 Beobachtungen (reine Python-Doppelschleife über
`n_x_candidates` Kandidaten × `n` Trainingswolken, kein Batching wie im
Original). Hochgerechnet auf `max_iteration=200` liegt eine einzelne
AIRBO-Optimierung in der Größenordnung von **Stunden statt Sekunden** – bei
500.000+ Kombinationen aktuell **nicht praktikabel** mit den Standardwerten.
Vor einem großen AIRBO-Lauf sollten `n_x_candidates`, `n_cloud_samples` und
`sub_samp_size` in `airbo.py` deutlich reduziert oder die MMD-Berechnung
vektorisiert werden.

## Ergebnisformat

`results.csv` ist im Long-Format (eine Zeile pro BO-Iteration) mit Spalten:
`task_id, combination_id, tensor_block, funktion, dimension,
surrogate_model, acquisition_model, sample_size_initial, max_iteration,
iteration, best_so_far, simple_regret, immediate_regret, x_distance,
status, error`. Fehlgeschlagene Tasks (`status="error"`) stoppen den Lauf
nicht, sondern werden mit Fehlermeldung protokolliert.
