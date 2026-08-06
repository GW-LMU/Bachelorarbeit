"""Fuegt alle Code-Unterordner von ROBO_Benchmark dem Python-Suchpfad hinzu.

Die Skripte sind nach Pipeline-Phase in Unterordner sortiert:

    Pre_Processing/   Tasks erzeugen und aufteilen (config.py, prepare_tasks.py, split_tasks.py)
    Main_Prozess/      Berechnung je LRZ-Instanz (core_bo.py, run_instance.py, check_progress.py)
    BO_Verfahren/       Surrogatmodell-Varianten (uncertainty_aware_bo.py, imprecise_gp.py, ...)
    Post_Processing/    Zusammenfuehren, Auswerten, Plotten (merge_results.py, ...)

Untereinander verwenden die Skripte weiterhin einfache imports wie
'import config' oder 'from imprecise_gp import ...' statt Paket-Importen
(z.B. 'from Pre_Processing.config import ...') - so bleiben sie unveraendert
lauffaehig, unabhaengig davon, aus welchem Unterordner heraus sie gestartet
werden. Jedes Skript mit Import aus einem Geschwisterordner beginnt daher mit:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pfade  # noqa: F401
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_UNTERORDNER = ["Pre_Processing", "Main_Prozess", "BO_Verfahren", "Post_Processing"]

for _name in _UNTERORDNER:
    _pfad = str(_ROOT / _name)
    if _pfad not in sys.path:
        sys.path.insert(0, _pfad)
