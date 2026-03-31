from __future__ import annotations

from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.viz import build_user_generation_figure, build_user_generation_plot_data
