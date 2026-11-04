"""Generate kaggle/pixel_artifacts_fp_v4_heads.ipynb (edit the cells here, not in the .ipynb).

    python kaggle/make_notebook.py
"""
import json
from pathlib import Path

INTRO = """# Pixel-Artifacts fp-v4.0 — fusion-head training and pillar ablations (Kaggle)

Inputs (Add Input):
1. **pixel-artifacts-feature-store** — `store_meta.json`, `stats.pt`, `*_X.npy`, `*_y.npy`, `*_copy.npy`, `*_index.csv`, `leakage_pairs.csv`
2. **pixel-artifacts-code** — the repository's `src/`, `scripts/`, `configs/`
3. *(resume only)* this notebook's previous output: finished runs are skipped, unfinished runs resume from their checkpoint.

Settings: GPU **T4 x2**, Internet **on** (installs `open_clip_torch`, which the fp-v3 module imports).
Run with **Save Version -> Save & Run All**. The runner stops itself after `HOURS`, so the output is saved before the 12 h limit."""

SETTINGS = """GROUP = "p3"      # p1 = full model, p2 = singles + leave-one-out, p3 = pairs + triples, all
PART = (2, 2)     # (k, n): this machine takes every n-th run of the group, starting at k
PROCS_PER_GPU = 1 # 1 process per T4: each needs ~4.5 GB RAM next to the 8.9 GB store
HOURS = 11.0"""

SETUP = """import glob, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
store = Path(glob.glob('/kaggle/input/**/store_meta.json', recursive=True)[0]).parent
work = Path('/kaggle/working'); runs = work / 'runs'; runs.mkdir(exist_ok=True)
found = glob.glob('/kaggle/input/**/scripts/train_heads.py', recursive=True)
if found:                                    # code dataset was unpacked by Kaggle
    shutil.copytree(Path(found[0]).parents[1], work / 'code', dirs_exist_ok=True)
else:                                        # code dataset kept as a zip
    zipfile.ZipFile(glob.glob('/kaggle/input/**/pixel-artifacts-code.zip', recursive=True)[0]).extractall(work / 'code')
code = work / 'code'
for prev in glob.glob('/kaggle/input/**/runs/*/', recursive=True):   # resume from an earlier version
    dst = runs / Path(prev).name
    if not (dst / 'DONE.json').exists():
        shutil.copytree(prev, dst, dirs_exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'open_clip_torch==3.3.0'], check=False)
print('store:', store, '| code:', code, '| runs present:', sorted(p.name for p in runs.iterdir()))
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                     capture_output=True, text=True).stdout)"""

TRAIN = """import torch
n_gpu = torch.cuda.device_count()
k, n = PART
slots = n_gpu * PROCS_PER_GPU
procs = []
for s in range(slots):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s % n_gpu), PIXEL_ARTIFACTS_WORKSPACE=str(work))
    log = open(work / f'train_slot{s + 1}.log', 'w')
    procs.append(subprocess.Popen(
        [sys.executable, '-u', str(work / 'code/scripts/train_heads.py'),
         '--group', GROUP, '--part', f'{k}/{n}', '--slot', f'{s + 1}/{slots}',
         '--hours', str(HOURS), '--store', str(store), '--runs', str(runs)],
        cwd=work / 'code', env=env, stdout=log, stderr=subprocess.STDOUT))
    print('slot', s + 1, 'of', slots, 'on gpu', s % n_gpu)
while any(p.poll() is None for p in procs):
    time.sleep(300)
    for s in range(slots):
        tail = (work / f'train_slot{s + 1}.log').read_text().strip().splitlines()[-1:]
        print(f'slot {s + 1}:', tail[0] if tail else '')
print('exit codes:', [p.returncode for p in procs])"""

SUMMARY = """import json
for d in sorted(runs.iterdir()):
    f = d / 'DONE.json'
    print(f'{d.name:45s}', json.loads(f.read_text()) if f.exists() else 'not finished: rerun to resume')"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", INTRO), cell("code", SETTINGS), cell("code", SETUP),
                cell("code", TRAIN), cell("code", SUMMARY)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("pixel_artifacts_fp_v4_heads.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
