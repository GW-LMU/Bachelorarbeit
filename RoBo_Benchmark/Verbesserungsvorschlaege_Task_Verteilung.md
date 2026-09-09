# Verbesserungsvorschläge: Task-Verteilung PC + VMs

Zusammenfassung der Chat-Diskussion vom 2026-08-27 zur geplanten Erweiterung
der Pipeline auf gemischte Rechenressourcen (1x lokaler PC mit 10 Kernen +
10x LRZ-VM mit je 48 Kernen).

## Ausgangslage / Ist-Zustand

- [`Pre_Processing/split_tasks.py`](Pre_Processing/split_tasks.py) teilt
  `tasks.csv` aktuell rein per Round-Robin auf: `task_id % n_instanzen == i`.
  Es gibt **keine** Logik, die nach Dimension oder Surrogatmodell filtert
  oder Instanzen unterschiedlich gewichtet.
- [`Post_Processing/merge_results.py`](Post_Processing/merge_results.py)
  führt beliebig viele `results.csv` per Glob-Pattern zusammen
  (`pd.concat`) – funktioniert bereits unabhängig von Ordnerstruktur/-namen,
  solange das Spaltenschema gleich ist.

## Vorschlag 1: Getrennte Task-Gruppen nach Dimension (PC vs. VMs)

**Ziel:** Der lokale PC (10 Kerne) rechnet ausschließlich niedrig-dimensionale
Probleme, die 10 VMs (je 48 Kerne) übernehmen die hoch-dimensionalen.

**Geht das mit der jetzigen Struktur?** Nein, nicht ohne Änderung – der
aktuelle Round-Robin-Split kennt keine Dimension.

**Umsetzung:**
1. `tasks.csv` weiterhin **einmal zentral** über `prepare_tasks.py` erzeugen
   (wichtig für eindeutige, durchgehende `task_id`s – siehe Vorschlag 2).
2. `split_tasks.py` um einen Dimension-Schwellenwert erweitern
   (z. B. `--dim-schwelle 10`), der `tasks.csv` zunächst in zwei Gruppen
   teilt:
   - "PC-Gruppe": `dimension <= dim_schwelle`
   - "VM-Gruppe": `dimension > dim_schwelle`
3. Die PC-Gruppe bleibt ein Chunk, die VM-Gruppe wird zusätzlich in 10
   (nahezu) gleich große Chunks aufgeteilt – idealerweise **gewichtet nach
   Kernzahl** statt stumpf gleich groß, da 10 Kerne (PC) vs. 10×48 Kerne
   (VMs) sehr unterschiedliche Kapazität bedeuten.
4. Ordnerstruktur z. B. `data/instances/pc/` und
   `data/instances/vm_0/ … vm_9/`.

## Vorschlag 2: Merge über beide Gruppen hinweg

**Frage:** Kommen am Ende zwei schematisch gleiche, aber getrennte
Ergebnisdateien raus, die man mergen muss?

**Antwort:** Ja – und das funktioniert bereits ohne Codeänderung, solange:
- beide Gruppen aus **derselben** `tasks.csv` stammen (ein zentraler
  `prepare_tasks.py`-Lauf), damit `task_id` über beide Gruppen hinweg
  eindeutig bleibt (keine doppelten IDs aus zwei unabhängigen Läufen mit
  je eigener Zählung ab 0),
- das Merge-Pattern beide Ordnerbäume erfasst, z. B.:

```bash
python Post_Processing/merge_results.py --pattern "data/instances/**/results/results.csv" --out data/ergebnisse_gesamt.xlsx
```

## Vorschlag 3: Task-Reihenfolge nach Dimension + Modell-Kosten sortieren

**Ziel:** PC und VMs sollen jeweils zuerst die einfachen/schnellen
Kombinationen abarbeiten und erst danach die teuren – ohne dass die
Ressourcen dafür komplett strikt getrennt werden müssen.

**Idee:** Die `task_id`-Vergabe bestimmt implizit die Bearbeitungsreihenfolge
(jede Instanz arbeitet ihre `tasks.csv` in Datei-Reihenfolge ab). Sortiert
man die Kombinationen **vor** der `task_id`-Vergabe nach

1. `dimension` (aufsteigend – niedrig-dimensional zuerst),
2. innerhalb jeder Dimension nach `surrogate_model` anhand einer
   Kostenrangfolge (billig → teuer, z. B. `GausianProzess` <
   `Imprecise_GausianProzess_Rodemann` < `Relevance Pursuit` <
   `STABLEOPT` < `DRBO` < `AIRBO`, siehe AIRBO-Laufzeitwarnung in der
   [README](README.md)),

dann bleibt dank Round-Robin-Split (`task_id % n_instanzen`) auf einer
bereits sortierten Liste jede resultierende Teilfolge pro Instanz selbst
grob aufsteigend sortiert. PC und VMs arbeiten so parallel von Anfang an
"erst easy, dann teuer" ab, ohne die Gruppen komplett trennen zu müssen –
kombinierbar mit Vorschlag 1.

**Umsetzung:**
- `config.py`: neue Liste `SURROGATE_KOSTEN_RANG = ["GausianProzess",
  "Imprecise_GausianProzess_Rodemann", "Relevance Pursuit", "STABLEOPT",
  "DRBO", "AIRBO"]`.
- `prepare_tasks.py::build_kombinationen()`: das entstandene DataFrame vor
  der `combination_id`-Vergabe nach `(dimension,
  kosten_rang[surrogate_model])` sortieren (bzw. `SURROGATE_AKQUISITION_PAARE`
  selbst in Kostenreihenfolge vorsortieren, `DIMENSIONEN` steht in der
  `itertools.product`-Reihenfolge bereits vor den Modell-Paaren).
- `build_tasks()` vergibt `task_id` automatisch in dieser Reihenfolge weiter.

**Offene Prüfung, bevor das umgesetzt wird:** Ob `run_instance.py` die
Tasks tatsächlich sequentiell in Datei-Reihenfolge abarbeitet, oder ob z. B.
mehrere Worker aus einer gemeinsamen Queue ohne Ordnungsgarantie ziehen /
die Reihenfolge shuffeln. Falls letzteres, müsste die Queue-Logik ebenfalls
angepasst werden, damit die Sortierung nicht wirkungslos verpufft.

## Nächste Schritte (offen, noch nicht umgesetzt)

- [ ] `split_tasks.py`: Dimension-Schwellenwert + PC/VM-Gruppenlogik
      (Vorschlag 1)
- [ ] `config.py` + `prepare_tasks.py`: Kostenrang-Sortierung der
      Kombinationen vor `task_id`-Vergabe (Vorschlag 3)
- [ ] `run_instance.py` prüfen: sequentielle Abarbeitung in Datei-Reihenfolge
      sicherstellen (Voraussetzung für Vorschlag 3)
- [ ] Merge-Aufruf in README/Pipeline-Skript auf `**`-Pattern anpassen
      (Vorschlag 2)
