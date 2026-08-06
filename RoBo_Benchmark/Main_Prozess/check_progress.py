"""Aggregierter Fortschritts-Ueberblick ueber mehrere Instanzen hinweg.

Liest die status.json-Dateien, die run_instance.py laufend schreibt.
Kann waehrend die Instanzen noch rechnen wiederholt aufgerufen werden.

    python Main_Prozess/check_progress.py --pattern "data/instances/instance_*/results/status.json"
    (aus der ROBO_Benchmark-Wurzel)
"""

import argparse
import glob
import json
from pathlib import Path


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Keine status.json gefunden fuer Pattern: {pattern}")
        return

    total_done = total_failed = total_all = 0
    print(f"{'Instanz':<20}{'erledigt':>10}{'gesamt':>10}{'%':>8}{'Tasks/s':>10}{'ETA (min)':>12}")
    for f in files:
        with open(f) as fh:
            s = json.load(fh)
        # status.json liegt unter .../instance_X/results/status.json
        name = Path(f).parents[1].name
        eta_min = s["eta_seconds"] / 60 if s.get("eta_seconds") else None
        eta_str = f"{eta_min:.1f}" if eta_min is not None else "-"
        print(f"{name:<20}{s['done']:>10}{s['total']:>10}{s['percent']:>7.1f}%{s['tasks_per_second']:>10.2f}{eta_str:>12}")
        total_done += s["done"]
        total_failed += s["failed"]
        total_all += s["total"]

    pct = 100 * total_done / total_all if total_all else 0
    print("-" * 70)
    print(f"GESAMT: {total_done}/{total_all} ({pct:.1f}%) | Fehlerhafte Tasks: {total_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()
    main(args.pattern)
