# Konzept: Verteilung auf AWS Spot-Instanzen

Zusammenfassung der Chat-Diskussion vom 2026-08-27 dazu, wie Code-Deployment
und Task-Verteilung aussehen könnten, falls statt/zusätzlich zu den
LRZ-Instanzen AWS Spot-Instanzen genutzt werden.

## Ausgangslage: passt das bestehende Design zu Spot?

[`Main_Prozess/run_instance.py`](Main_Prozess/run_instance.py) ist bereits
**idempotent/resumable**: beim Start wird `results.csv` gelesen und nur
`task_id`s mit `status="ok"` übersprungen
([run_instance.py:119-126](Main_Prozess/run_instance.py:119)). Genau diese
Eigenschaft braucht man für Spot, da Instanzen jederzeit mit 2 Minuten
Vorwarnung reclaimed werden können.

**Der wunde Punkt:** `results.csv` liegt nur lokal auf der Instanz
(`output_dir`). Stirbt die Spot-Instanz, ist die lokale Disk weg – dann
bringt "resumable" nichts, weil es nichts zum Fortsetzen mehr gibt. Das ist
der Kernpunkt, der vor einem produktiven Spot-Einsatz gelöst werden muss
(siehe Abschnitt 3).

## 1) Code hochladen

Zwei Ansätze, je nachdem wie oft sich der Code ändert:

**Custom AMI (empfohlen)** – `cocoex` ist eine C-Extension, `pip install`
(inkl. Kompilierung) kostet auf jeder frischen Instanz spürbar Zeit:
1. Einmal eine EC2-Instanz hochfahren, Repo klonen,
   `pip install -r requirements.txt` (inkl. `cocoex`-Build) durchlaufen
   lassen.
2. Daraus ein AMI erstellen (`aws ec2 create-image`).
3. Jede Spot-Instanz startet direkt mit diesem AMI – Code + alle
   Dependencies sind sofort da, kein Boot-Overhead.

Bei Codeänderungen: entweder neues AMI bauen (größere Änderungen), oder im
User-Data-Script beim Boot ein `git pull` im schon vorhandenen
Repo-Ordner laufen lassen (kleine Änderungen – schnell, kein Neu-Build der
Dependencies nötig).

**Alternative ohne AMI:** User-Data-Script macht bei jedem Boot
`git clone` + `pip install` – einfacher aufzusetzen, aber jeder Spot-Start
dauert durch den `cocoex`-Build spürbar länger (mehrere Minuten Overhead ×
jede Instanz, die reclaimed und neu gestartet wird).

## 2) Tasks verteilen

Die bestehende `data/instances/instance_<i>/`-Struktur aus
[`split_tasks.py`](Pre_Processing/split_tasks.py) passt 1:1 weiter – nur der
Transportweg ändert sich von `scp` zu **S3**:

```bash
# lokal, einmalig (wie bisher):
python Pre_Processing/prepare_tasks.py
python Pre_Processing/split_tasks.py --instanzen 10

# je Chunk nach S3 hochladen statt scp:
aws s3 cp data/instances/instance_0 s3://<bucket>/instance_0 --recursive
...
```

Im User-Data-Script jeder Spot-Instanz (die z. B. per Tag/Launch-Template
weiß, welcher Chunk-Index ihr gehört):

```bash
aws s3 cp s3://<bucket>/instance_$INDEX /opt/robo/data/instances/instance_$INDEX --recursive
python Main_Prozess/run_instance.py --instance-dir data/instances/instance_$INDEX --workers 48
```

## 3) Das kritische Stück: Ergebnisse überleben lassen

Da Spot-Instanzen jederzeit weg sein können, **darf `results.csv` nicht nur
lokal liegen**. Zwei Optionen:

- **Periodischer S3-Sync** (einfach): ein Cronjob/Hintergrundprozess auf der
  Instanz macht alle 1–2 Minuten
  `aws s3 sync results/ s3://<bucket>/instance_$INDEX/results/`. Bei
  Reclaim gehen höchstens die letzten 1–2 Minuten verloren – durch die
  vorhandene Resume-Logik unproblematisch, die Instanz (bzw. ihr
  Nachfolger) macht dort weiter, wo `results.csv` aufgehört hat.
- **EFS-Mount statt lokalem `output_dir`** (robuster, etwas mehr Setup):
  `--output-dir` zeigt direkt auf ein gemountetes EFS-Verzeichnis. Dann
  gibt's gar keinen Sync-Schritt und keinen Datenverlustzeitraum – kostet
  aber laufend EFS-Gebühren und etwas mehr I/O-Latenz pro Schreibzugriff
  (bei `flush()` nach jedem Task, siehe
  [run_instance.py:206](Main_Prozess/run_instance.py:206), relevant bei
  sehr vielen kleinen Tasks).

Bei einer Spot-Neubeschaffung nach Interrupt: einfach denselben
`instance_$INDEX`-Ordner (inkl. der aus S3/EFS wiederhergestellten
`results.csv`) neu bereitstellen und `run_instance.py` erneut starten –
läuft automatisch dank vorhandener Resume-Logik weiter.

## 4) Orchestrierung

Für 10 Instanzen reicht ein einfacher **EC2 Spot Fleet** (oder eine Auto
Scaling Group mit Spot-Kapazität) mit dem AMI aus Abschnitt 1 und einem
instanzspezifischen User-Data-Script.

Alternativ, wenn's komfortabler sein soll: **AWS Batch mit
Spot-Compute-Environment** – dort übernimmt AWS automatisch das
Nachstarten unterbrochener Jobs aus einer Queue, spart das manuelle
"Chunk-Index pro Instanz zuweisen". Würde aber bedeuten, den
Instanz-Chunk-Ansatz auf eine Batch-Job-pro-Chunk-Struktur umzubauen – mehr
Umbauaufwand, aber weniger Betriebsaufwand danach.

## Nächste Schritte (offen, noch nicht umgesetzt)

- [ ] AMI bauen (Repo + Dependencies inkl. `cocoex`)
- [ ] S3-Bucket + Upload-Workflow für `instance_<i>`-Chunks
- [ ] Sync-Mechanismus für `results.csv` (S3-Sync-Cron oder EFS-Mount)
- [ ] User-Data-Script je Instanz (Chunk-Download, `run_instance.py`-Start)
- [ ] Entscheidung: einfacher Spot Fleet vs. AWS Batch mit
      Spot-Compute-Environment
- [ ] Merge-Schritt (`merge_results.py`) nach Lauf auf die aus S3/EFS
      zusammengeführten `results.csv` aller Instanzen anwenden
