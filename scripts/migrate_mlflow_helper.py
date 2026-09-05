"""Replace the copy-pasted MLflow helper cell with the shared module import.

Usage: uv run python scripts/migrate_mlflow_helper.py [--check]
"""

from __future__ import annotations

import json
import sys

FILES = [
    "experiments/image_classification/resnet18/resnet18_cifar10.ipynb",
    "experiments/image_classification/mobilenetv2/mobilenetv2_cifar10.ipynb",
    "experiments/vit/vit_cifar10/vit_cifar10.ipynb",
    "experiments/vit/mae/mae_cifar10.ipynb",
    "experiments/vit/dino/dino_cifar10.ipynb",
]

NEW_SOURCE = [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    from _mlflow import make_logger\n",
    "except ImportError:  # kernel cwd is the experiment dir, not repo root\n",
    "    for _base in [Path.cwd(), *Path.cwd().parents]:\n",
    '        if (_base / "_mlflow.py").exists():\n',
    "            sys.path.insert(0, str(_base))\n",
    "            break\n",
    '        if (_base / "experiments" / "_mlflow.py").exists():\n',
    '            sys.path.insert(0, str(_base / "experiments"))\n',
    "            break\n",
    "    from _mlflow import make_logger\n",
    "\n",
    "logger = make_logger(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ROOT / \"mlruns\", enabled=MLFLOW_ENABLED)\n",
]


def main() -> None:
    check = "--check" in sys.argv
    for f in FILES:
        nb = json.load(open(f, encoding="utf-8"))
        hits = [
            c for c in nb["cells"]
            if c["cell_type"] == "code"
            and "def make_logger" in (''.join(c["source"]) if isinstance(c["source"], list) else c["source"])
        ]
        assert len(hits) == 1, f"{f}: found {len(hits)} helper cells"
        cell = hits[0]
        if cell["source"] == NEW_SOURCE:
            print(f"OK   {f} (already migrated)")
            continue
        if check:
            print(f"TODO {f}")
            continue
        cell["source"] = list(NEW_SOURCE)
        cell["outputs"] = []
        cell["execution_count"] = None
        json.dump(nb, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"MIGRATED {f}")


if __name__ == "__main__":
    main()
