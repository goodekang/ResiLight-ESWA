from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(name):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_default(overrides=None):
    cfg = load_yaml("default.yaml")
    if overrides:
        _merge(cfg, overrides)
    return cfg


def _merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_modes():
    return load_yaml("modes.yaml")


def load_seeds():
    return load_yaml("seeds.yaml")
