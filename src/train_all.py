
import subprocess, sys, time
from pathlib import Path

src_dir      = Path(__file__).resolve().parent
project_dir  = src_dir.parent
SCRIPTS = [
    src_dir / "train_atten_single.py",
    src_dir / "train_atten_multi.py"
    # src_dir/ "train_without.py",
]

for script in SCRIPTS:
    if not script.exists():
        raise FileNotFoundError(f"can't find：{script}")
    print(f"\n start training {script.name} ...")

    ret = subprocess.run([sys.executable, str(script)], cwd=project_dir)
    if ret.returncode:
        raise RuntimeError(f"{script.name} excit code {ret.returncode}")
    time.sleep(30)
print("\n all train completed!")
