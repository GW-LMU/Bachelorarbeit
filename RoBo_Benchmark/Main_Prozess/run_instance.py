"""Laeuft auf JEDER der 4 LRZ-Instanzen und rechnet ihren Task-Chunk parallel ab.

Aufruf, z.B. auf instance_0 mit 10 Kernen (aus der ROBO_Benchmark-Wurzel):

    python Main_Prozess/run_instance.py --instance-dir data/instances/instance_0 --workers 10

Fortschritt:
    - laufender tqdm-Balken auf der Konsole dieser Instanz
    - status.json im Ergebnisordner wird laufend aktualisiert
      (kann ueber check_progress.py instanzuebergreifend ausgewertet werden,
      sobald die status.json-Dateien an einem Ort gesammelt sind)

Wiederaufnahme nach Absturz/Neustart:
    - erfolgreich erledigte Tasks stehen in results.csv (status=ok) und
      werden beim naechsten Start automatisch uebersprungen
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) und BO_Verfahren/ zu sys.path hinzu

import argparse
import csv
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cocoex

import config
import core_bo

RESULT_COLUMNS = [
    "task_id", "combination_id", "tensor_block", "funktion", "dimension",
    "surrogate_model", "acquisition_model", "sample_size_initial",
    "max_iteration", "iteration", "best_so_far", "simple_regret",
    "immediate_regret", "x_distance", "status", "error",
]

_WORKER_STATE = {}


def _init_worker(tensor_path, bounds):
    # Eigenes Arbeitsverzeichnis pro Prozess: cocoex schreibt bei der
    # Optimum-Abfrage eine Datei mit festem Namen ins cwd - ohne eigenes
    # cwd wuerden sich parallele Worker gegenseitig ueberschreiben.
    worker_dir = Path(tempfile.mkdtemp(prefix=f"bo_worker_{os.getpid()}_"))
    os.chdir(worker_dir)

    _WORKER_STATE["suite"] = cocoex.Suite("bbob", "", "")
    _WORKER_STATE["tensor_noise"] = np.load(tensor_path)
    _WORKER_STATE["bounds"] = bounds


def _run_one_task(task_row):
    suite = _WORKER_STATE["suite"]
    tensor_noise = _WORKER_STATE["tensor_noise"]
    bounds = _WORKER_STATE["bounds"]
    try:
        return core_bo.run_task(task_row, suite, tensor_noise, bounds)
    except Exception as exc:  # ein fehlerhafter Task darf den Lauf nicht stoppen
        return [{
            **task_row,
            "iteration": -1,
            "best_so_far": None,
            "simple_regret": None,
            "immediate_regret": None,
            "x_distance": None,
            "status": "error",
            "error": str(exc),
        }]


def load_done_task_ids(results_path):
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path, usecols=["task_id", "status"])
    return set(df.loc[df["status"] == "ok", "task_id"].unique())


def write_status(status_path, done, total, failed, start_time):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta_seconds = remaining / rate if rate > 0 else None
    status = {
        "done": done,
        "failed": failed,
        "total": total,
        "percent": round(100 * done / total, 2) if total else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "tasks_per_second": round(rate, 3),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = status_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)
    tmp_path.replace(status_path)


def main(instance_dir, workers, output_dir):
    instance_dir = Path(instance_dir)
    tasks_path = instance_dir / "tasks.csv"
    # absolut, da die Worker-Prozesse ihr cwd wechseln (siehe _init_worker)
    tensor_path = (instance_dir / "tensor_noise.npy").resolve()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    status_path = output_dir / "status.json"

    df_tasks = pd.read_csv(tasks_path)
    done_ids = load_done_task_ids(results_path)
    pending = df_tasks[~df_tasks["task_id"].isin(done_ids)]

    total = len(df_tasks)
    print(f"Tasks gesamt: {total} | bereits erledigt: {len(done_ids)} | offen: {len(pending)}")

    if len(pending) == 0:
        write_status(status_path, len(done_ids), total, 0, time.time())
        print("Nichts zu tun - alle Tasks bereits erledigt.")
        return

    bounds = (config.MINUS_AREA, config.PLUS_AREA)

    file_exists = results_path.exists()
    results_file = open(results_path, "a", newline="")
    writer = csv.DictWriter(results_file, fieldnames=RESULT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    start_time = time.time()
    done = len(done_ids)
    failed = 0

    task_dicts = pending.to_dict("records")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(tensor_path, bounds),
    ) as pool:
        futures = [pool.submit(_run_one_task, t) for t in task_dicts]

        with tqdm(total=len(futures), desc=f"instance {instance_dir.name}") as pbar:
            for future in as_completed(futures):
                rows = future.result()
                for row in rows:
                    writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})
                results_file.flush()

                if rows and rows[0].get("status") == "error":
                    failed += 1
                done += 1
                pbar.update(1)

                if done % 5 == 0 or done == total:
                    write_status(status_path, done, total, failed, start_time)

    write_status(status_path, done, total, failed, start_time)
    results_file.close()
    print(f"Fertig. Ergebnisse in: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", required=True, help="Ordner mit tasks.csv + tensor_noise.npy")
    parser.add_argument("--workers", type=int, default=config.WORKERS_PRO_INSTANZ)
    parser.add_argument("--output-dir", default=None, help="Standard: <instance-dir>/results")
    args = parser.parse_args()

    output_dir = args.output_dir or (Path(args.instance_dir) / "results")
    main(args.instance_dir, args.workers, output_dir)
