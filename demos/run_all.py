"""Run demos 01-08 in order (demo 00 only regenerates the included trip file).

    python demos/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in sorted(HERE.glob("0[1-9]_*.py")):
    t = time.perf_counter()
    print(f"--- {script.name}", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)
    print(f"    {time.perf_counter() - t:.0f} s", flush=True)
