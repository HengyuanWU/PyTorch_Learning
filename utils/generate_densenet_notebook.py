#!/usr/bin/env python
"""create_densenet_notebook.py
---------------------------------------------------------------------
Generate **densenet.ipynb** that mirrors all the explorations
from your LeNet notebook:

* Baseline runs for the two DenseNet variants you already coded
  (a lightweight CIFAR‑DenseNet and the full DenseNet‑121).
* A **grid‑search hyper‑parameter sweep** (batch‑size × lr × optimiser)
  just like the LeNet notebook did.
* Automatic selection & visualisation of the best run.

All heavy lifting stays in **densenet.py** and **train_densenet.py** –
this notebook only imports and orchestrates.

Usage
-----
```bash
python create_densenet_notebook.py   # (re)generates the notebook
jupyter lab densenet.ipynb  # open & run cells step‑by‑step
```

Dependencies: `pip install nbformat torch torchvision matplotlib tqdm`.
"""

from pathlib import Path
import nbformat as nbf

# ──────────────────────────────────────────────────────────────────────────
# Helper shortcuts
# ──────────────────────────────────────────────────────────────────────────

def c(code: str):
    """Return a *code* cell."""
    return nbf.v4.new_code_cell(code)


def m(md: str):
    """Return a *markdown* cell."""
    return nbf.v4.new_markdown_cell(md)


# ──────────────────────────────────────────────────────────────────────────
# Assemble notebook cells
# ──────────────────────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb["cells"] = [
    # ─────────────────────────────────────────────────────────────── 1 ──
    m("""# 🧬 DenseNet Exploration Notebook\n
Welcome!  This notebook re‑creates all the learning experiments you did
with **LeNet** – but for **DenseNet**.  We reuse your existing
`densenet.py` model definitions and the training helper
`train_densenet.py` so you can focus on experimentation rather than
boilerplate.
"""),

    # ─────────────────────────────────────────────────────────────── 2 ──
    m("""## Setup"""),
    c("""# Uncomment on first run\n# !pip install --quiet torch torchvision matplotlib tqdm\n\nfrom train_densenet import run_densenet_training  # 👈 your helper\nfrom itertools import product\nimport json, time, pathlib, matplotlib.pyplot as plt\n"""),

    # ─────────────────────────────────────────────────────────────── 3 ──
    m("""## 1️⃣  Baseline: lightweight CIFAR‑DenseNet"""),

    c("""history_light = run_densenet_training(\n    model_type='densenetcustom',   # lightweight variant\n    epochs=100,\n    train_batch_size=128,\n    test_batch_size=256,\n    learning_rate=0.1,\n    optimiser='sgd',\n)\n"""),

    m("""### Learning curves (lightweight)"""),
    c("""val_loss, val_acc = zip(*history_light)\nplt.figure(figsize=(6,4))\nplt.plot(val_loss); plt.title('Light DenseNet – validation loss');\nplt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()\n\nplt.figure(figsize=(6,4))\nplt.plot(val_acc); plt.title('Light DenseNet – validation accuracy');\nplt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()\n"""),

    # ─────────────────────────────────────────────────────────────── 4 ──
    m("""## 2️⃣  Baseline: full DenseNet‑121"""),
    c("""history_121 = run_densenet_training(\n    model_type='densenet121',\n    epochs=100,\n    train_batch_size=128,\n    test_batch_size=256,\n    learning_rate=0.1,\n    optimiser='sgd',\n)\n"""),

    m("""### Learning curves (DenseNet‑121)"""),
    c("""val_loss, val_acc = zip(*history_121)\nplt.figure(figsize=(6,4))\nplt.plot(val_loss); plt.title('DenseNet‑121 – validation loss');\nplt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()\n\nplt.figure(figsize=(6,4))\nplt.plot(val_acc); plt.title('DenseNet‑121 – validation accuracy');\nplt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()\n"""),

    # ─────────────────────────────────────────────────────────────── 5 ──
    m("""## 3️⃣  Grid‑search hyper‑parameter sweep"""),
    m("""We replicate the grid‑search you performed with LeNet.  The search
space below is small by default so the notebook finishes in a reasonable
time – feel free to expand the lists."""),

    c("""param_grid = {\n    'model_type':        ['densenetcustom', 'densenet121'],\n    'train_batch_size':  [64, 128],\n    'learning_rate':     [0.1, 0.01],\n    'optimiser':         ['sgd', 'adam'],\n}\n\nsearch_results = []\nrun_id = 0\n\nfor model_type, bs, lr, opt in product(\n        param_grid['model_type'],\n        param_grid['train_batch_size'],\n        param_grid['learning_rate'],\n        param_grid['optimiser']):\n    run_id += 1\n    print(f"\n🔍  Run {run_id}: {model_type}, bs={bs}, lr={lr}, opt={opt}")\n    hist = run_densenet_training(\n        model_type=model_type,\n        epochs=50,                        # shorter epochs for search\n        train_batch_size=bs,\n        test_batch_size=256,\n        learning_rate=lr,\n        optimiser=opt,\n        silent=True,                       # assume helper supports this\n    )\n    best_acc = max(acc for _loss, acc in hist)\n    search_results.append({\n        'model_type': model_type, 'batch_size': bs, 'lr': lr,\n        'optimiser': opt, 'best_val_acc': best_acc\n    })\n\nprint("\n✅  Grid search complete!\n")\n"""),

    m("""### Results DataFrame"""),
    c("""import pandas as pd\nres_df = pd.DataFrame(search_results)\nres_df.sort_values('best_val_acc', ascending=False, inplace=True)\nres_df.reset_index(drop=True, inplace=True)\nres_df\n"""),

    m("""### Best configuration"""),
    c("""best_cfg = res_df.iloc[0]\nprint(best_cfg)\n"""),

    # ─────────────────────────────────────────────────────────────── 6 ──
    m("""## 4️⃣  Train best DenseNet from scratch"""),
    c("""history_best = run_densenet_training(\n    model_type=best_cfg.model_type,\n    epochs=150,\n    train_batch_size=int(best_cfg.batch_size),\n    test_batch_size=256,\n    learning_rate=float(best_cfg.lr),\n    optimiser=best_cfg.optimiser,\n)\n"""),

    m("""### Final curves & insights"""),
    c("""val_loss, val_acc = zip(*history_best)\nplt.figure(figsize=(6,4))\nplt.plot(val_loss); plt.title('Best DenseNet – validation loss');\nplt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()\n\nplt.figure(figsize=(6,4))\nplt.plot(val_acc); plt.title('Best DenseNet – validation accuracy');\nplt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()\n"""),

    # ─────────────────────────────────────────────────────────────── 7 ──
    m("""## 5️⃣  Next explorations"""),
    m("""* Try **DenseNet‑BC** variants, vary growth‑rate *k* & compression θ.\n* Add **cutout / RandAugment** to boost generalisation.\n* Replace the classifier with **ArcFace** head for metric learning
  experiments.\n* Port the best‑found config onto **Tiny‑Imagenet** to see scaling
  behaviour.
"""),
]

# ──────────────────────────────────────────────────────────────────────────
# Write notebook file
# ──────────────────────────────────────────────────────────────────────────

nb_path = Path("../notebooks/densenet.ipynb")
nbf.write(nb, nb_path.open("w", encoding="utf-8"))
print(f"\n✅  Notebook written to {nb_path.resolve()}")
