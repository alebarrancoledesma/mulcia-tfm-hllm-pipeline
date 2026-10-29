"""Familias CVRP, splits (prompt/dev/test) y resolución de rutas PDDL."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = Path(
    os.environ.get("TFM_ARTIFACTS_DIR", PROJECT_ROOT.parent / "mulcia-tfm-artifacts")
).resolve()
NUMERIC_INPUTS = ARTIFACTS_ROOT / "inputs" / "02-numeric-planning"
PDDL_ROOT = NUMERIC_INPUTS / "02-pddl-converted"
SPLITS_ROOT = NUMERIC_INPUTS / "03-splits"

VALID_SPLITS = ("prompt", "dev", "test")


def domain_path():
    return PDDL_ROOT / "domain.pddl"


def instance_path(family, instance_id):
    return PDDL_ROOT / family / f"{instance_id}.pddl"


def instance_paths(family, instance_ids):
    return [instance_path(family, instance_id) for instance_id in instance_ids]


def split_path(family, split):
    if split not in VALID_SPLITS:
        raise ValueError(f"Split desconocido '{split}' (válidos: {VALID_SPLITS})")
    return SPLITS_ROOT / family / f"{split}.json"


def load_split(family, split):
    """Lista de instance_ids del manifiesto prompt/dev/test."""
    with open(split_path(family, split), "r", encoding="utf-8") as f:
        return json.load(f)["instances"]


def load_split_manifest(family, split):
    with open(split_path(family, split), "r", encoding="utf-8") as f:
        return json.load(f)


def list_families():
    if not SPLITS_ROOT.exists():
        return []
    return sorted(p.name for p in SPLITS_ROOT.iterdir() if p.is_dir())


def assert_no_overlap(family):
    """Comprueba que prompt/dev/test no comparten instancias (anti-leakage)."""
    prompt = set(load_split(family, "prompt"))
    dev = set(load_split(family, "dev"))
    test = set(load_split(family, "test"))
    overlap = (prompt & dev) | (prompt & test) | (dev & test)
    if overlap:
        raise ValueError(f"Leakage en familia '{family}': {sorted(overlap)}")
