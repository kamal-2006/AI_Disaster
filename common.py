from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_project_root_on_path() -> Path:
    """Make sure imports work when scripts are executed from the ai/ folder."""

    root = PROJECT_ROOT
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def resolve_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
