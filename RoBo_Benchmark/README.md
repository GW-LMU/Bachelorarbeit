# ROBO_Benchmark – Verteilte BO-Benchmark-Pipeline

Rechnet die BO-Benchmarks auf mehreren unabhaengigen LRZ-Instanzen parallel.
Eine Task = eine (Parameterkombination × Tensorblock)-Kombination; nur die
Iterationsschleife innerhalb einer Task ist sequentiell, alles andere laeuft
parallel. Alle Befehle unten werden **aus der ROBO_Benchmark-Wurzel** heraus
ausgefuehrt (nicht aus den Unterordnern).

## Architektur

Die Skripte sind nach Pipeline-Phase in vier Unterordner sortiert:

```
ROBO_Benchmark/
├── pfade.py             # Bootstrap: haengt die Unterordner unten in sys.path ein
├── config.py? nein →     (liegt in Pre_Processing/, siehe unten)
│
├── Pre_Processing/       # 1) Tasks erzeugen und aufteilen
│   ├── config.py            zentrale Konfiguration (Funktionen, Dimensionen, Modelle, Seeds, ...)
│   ├── prepare_tasks.py      erzeugt EINMALIG data/tasks.csv + tensor_noise.npy
│   └── split_tasks.py        teilt tasks.csv auf N Instanz-Ordner auf
│
├── Main_Prozess/         # 2) Berechnung je LRZ-Instanz
│   ├── core_bo.py            BO-Kernfunktionen + Varianten-Dispatch (select_and_run_variant)
│   ├── run_instance.py       laeuft auf jeder Instanz, rechnet den Task-Chunk parallel ab
│   └── check_progress.py     aggregierter Fortschritts-Ueberblick ueber alle Instanzen
│
├── BO_Verfahren/         # die austauschbaren Surrogatmodell-Varianten
│   ├── uncertainty_aware_bo.py   BayesOptimizer-Kern fuer imprecise_gp.py
│   ├── imprecise_gp.py           Imprecise GP (Rodemann), nur dim=1
│   ├── relevance_pursuit.py      Robuste Relevance-Pursuit-GP (Ament et al. 2024)
│   ├── stable_opt.py             STABLEOPT (Bogunovic et al. 2018)
│   ├── drbo.py                   Distributionally Robust BO (Kirschner et al. 2020)
│   └── airbo.py                  AIRBO (Yang et al. 2023)
│
├── Post_Processing/      # 3) Zusammenfuehren, Auswerten, Plotten
│   ├── merge_results.py               fuehrt alle results.csv zu einer Gesamtdatei zusammen
│   ├── pivot_simple_regret.py         pivotiert ins Wide-Format (Var_1..Var_n)
│   ├── compute_confidence_interval.py ersetzt Var_1..Var_n durch Mittelwert + Konfidenzintervall
│   └── plot_convergence.py            filtert + plottet die Konvergenzgeschwindigkeit
│
├── KEY_LRZ/              # Zugangsdaten fuer die LRZ-Instanzen - NICHT in Git, siehe Warnung unten
├── data/                 # generierte Zwischen-/Ergebnisdateien, siehe Tabelle unten
└── requirements.txt
```

**Warum diese Aufteilung?** Die Skripte verwenden weiterhin einfache imports
wie `import config` oder `from imprecise_gp import ...` statt Paket-Importen
(`from Pre_Processing.config import ...`) - unveraendert gegenueber der
vorherigen flachen Ablage. Damit das trotz der Unterordner funktioniert,
haengt `pfade.py` beim Start jedes Skripts automatisch alle vier Unterordner
in den Python-Suchpfad ein (siehe die ersten Zeilen jeder Datei mit
Cross-Ordner-Import). Das ist reine Bootstrap-Logik, an der Skript-Logik hat
sich nichts geaendert.

### ⚠️ KEY_LRZ/ nicht committen

`KEY_LRZ/KEX.txt` enthaelt einen privaten SSH-Schluessel fuer den LRZ-Zugriff.
Der Ordner ist in `.gitignore` (Projektwurzel) eingetragen und darf **nicht**
in dieses (oeffentliche) Git-Repository gelangen.

## Ablauf

**1) Einmalig zentral (z.B. auf deinem Rechner):**

```bash
python Pre_Processing/prepare_tasks.py               # erzeugt data/tasks.csv, data/tensor_noise.npy, data/kombinationen.csv
python Pre_Processing/split_tasks.py --instanzen 4    # teilt in data/instances/instance_0..3 auf
```

**2) Pro Instanz: `data/instances/instance_<i>/` (enthaelt `tasks.csv` +
`tensor_noise.npy`) auf die jeweilige Instanz kopieren, z.B.:**

```bash
scp -r data/instances/instance_0 user@instanz0:~/robo_benchmark/data/instances/instance_0
```

Fuer die eigentliche Berechnung auf der LRZ-Instanz braucht es zusaetzlich
den Code: `Main_Prozess/`, `BO_Verfahren/`, `Pre_Processing/config.py` und
`pfade.py` (z.B. per `rsync` den ganzen Ordner ohne `KEY_LRZ/` und `data/`
kopieren).

**3) Auf jeder Instanz die Berechnung starten:**

```bash
python Main_Prozess/run_instance.py --instance-dir data/instances/instance_0 --workers 10
```

Läuft der Prozess ab (Absturz, Neustart, SSH-Abbruch), einfach denselben
Befehl erneut ausfuehren – bereits erledigte Tasks werden anhand von
`results.csv` automatisch übersprungen.

## Fortschritt verfolgen

- Auf jeder Instanz läuft ein `tqdm`-Balken direkt in der Konsole.
- Jede Instanz schreibt laufend `results/status.json` (erledigt/gesamt/%,
  Tasks pro Sekunde, geschätzte Restzeit).
- Sobald die `status.json`-Dateien aller Instanzen an einem Ort verfügbar
  sind (z.B. per `scp` zurückkopiert oder gemountet), aggregierter Überblick:

```bash
python Main_Prozess/check_progress.py --pattern "data/instances/instance_*/results/status.json"
```

Kann beliebig oft während der laufenden Berechnung ausgeführt werden.

## Ergebnisse zusammenführen und auswerten

Nach Abschluss aller Instanzen (`results.csv` je Instanz zurückkopiert):

```bash
python Post_Processing/merge_results.py --pattern "data/instances/instance_*/results/results.csv" --out data/ergebnisse_gesamt.xlsx
python Post_Processing/pivot_simple_regret.py --pattern "data/instances/instance_*/results/results.csv" --out data/ergebnisse_wide.xlsx
python Post_Processing/compute_confidence_interval.py --in data/ergebnisse_wide.xlsx --out data/ergebnisse_konfidenzintervall.xlsx
python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --funktion 1 --dimension 2 --out plots/f1_d2.png
```

`compute_confidence_interval.py` nimmt die Wide-Format-Datei von
`pivot_simple_regret.py` (eine Zeile je Kombination+Iteration, Tensorblöcke
nebeneinander in `Var_1..Var_n`) und ersetzt die `Var_n`-Spalten durch
`mean`, `std`, `ci_lower`, `ci_upper`, `ci_halfwidth` (t-Verteilung, Default
95 % Konfidenzniveau, per `--konfidenz` änderbar) — wieder im Wide-Format,
eine Zeile je Kombination+Iteration.

`plot_convergence.py` filtert diese Datei ueber `--funktion`/`--dimension`/
`--surrogate-model`/`--acquisition-model` (jeder Parameter optional) und
zeichnet `mean` über `iteration` als Linie, `ci_lower`/`ci_upper` als
Konfidenzband. Bleiben nach dem Filtern mehrere Kombinationen übrig (z.B.
mehrere Surrogatmodelle fuer dieselbe Funktion/Dimension), wird pro
Kombination eine eigene Linie + Band gezeichnet — direkter
Konvergenzgeschwindigkeits-Vergleich mehrerer BO-Varianten.

## Datendateien in data/

| Datei | Erzeugt von | Inhalt |
|---|---|---|
| `kombinationen.csv` | prepare_tasks.py | eine Zeile je Parameterkombination (Kreuzprodukt aus config.py) |
| `tasks.csv` | prepare_tasks.py | eine Zeile je (Kombination × Tensorblock) = eine Task |
| `tensor_noise.npy` | prepare_tasks.py | Stichproben-Tensor fuer die Initialsamples aller Tensorbloecke |
| `instances/instance_<i>/` | split_tasks.py | Task-Anteil + Tensor-Kopie je Instanz |
| `instances/instance_<i>/results/results.csv` | run_instance.py | Long-Format-Ergebnis dieser Instanz (eine Zeile je BO-Iteration) |
| `instances/instance_<i>/results/status.json` | run_instance.py | laufender Fortschritt dieser Instanz |
| `ergebnisse_gesamt.xlsx` | merge_results.py | alle instanz-`results.csv` zusammengefuehrt (Long-Format) |
| `ergebnisse_wide.xlsx` | pivot_simple_regret.py | Wide-Format, `simple_regret` je Tensorblock in `Var_1..Var_n` |
| `ergebnisse_konfidenzintervall.xlsx` | compute_confidence_interval.py | Wide-Format, `Var_n` ersetzt durch Mittelwert + Konfidenzintervall |

## Konfiguration

Alle Parameter (Funktionen, Dimensionen, Modelle, Iterationen, Seeds, Anzahl
Instanzen/Worker) liegen zentral in `Pre_Processing/config.py`. Für den
Zielumfang von 500.000+ Kombinationen dort `FUNKTIONEN`, `DIMENSIONEN`,
`SURROGATE_AKQUISITION_PAARE`, `N_SAMPLE_INIT` entsprechend erweitern.

## Implementierte Surrogatmodell-Varianten

Analog zu `waehle_und_berechne_variante()` in `ROBO/coco_benchmark.py` sind
in `core_bo.select_and_run_variant()` folgende Varianten angebunden:

| Surrogatmodell (`config.SURROGATE_AKQUISITION_PAARE`) | Akquisition | Skript | Einschränkung |
|---|---|---|---|
| `"GausianProzess"` | `EI` \| `UCB` \| `LCB` | `core_bo.py` (`run_gaussian_process`) | — |
| `"Imprecise_GausianProzess_Rodemann"` | `LCB` \| `GLCB` | `BO_Verfahren/imprecise_gp.py` + `uncertainty_aware_bo.py` | nur `dim=1` |
| `"Relevance Pursuit"` | `EI` \| `UCB` \| `LCB` | `BO_Verfahren/relevance_pursuit.py` | braucht neuere `botorch`-Version |
| `"STABLEOPT"` | nur `UCB` | `BO_Verfahren/stable_opt.py` | feste (ucb,lcb)-Konfidenzbänder intern |
| `"DRBO"` | nur `"distributionally robust UCB-Akquisition"` | `BO_Verfahren/drbo.py` | letzte Spalte von `dim` = Kontextvariable (`context_dim=1` fest) |
| `"AIRBO"` | nur `UCB` | `BO_Verfahren/airbo.py` | ⚠️ quadratische Laufzeit, siehe Warnung unten |

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

`results.csv` (und `ergebnisse_gesamt.xlsx`) sind im Long-Format (eine Zeile
pro BO-Iteration) mit Spalten: `task_id, combination_id, tensor_block,
funktion, dimension, surrogate_model, acquisition_model, sample_size_initial,
max_iteration, iteration, best_so_far, simple_regret, immediate_regret,
x_distance, status, error`. Fehlgeschlagene Tasks (`status="error"`) stoppen
den Lauf nicht, sondern werden mit Fehlermeldung protokolliert.
