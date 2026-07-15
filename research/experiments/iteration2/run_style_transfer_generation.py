#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
BASE_PATH = REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py"
PAYLOAD_PATH = Path(__file__).with_name("style_transfer_payloads.py")


def requested_experiment_root() -> Path | None:
    for index, value in enumerate(sys.argv):
        if value == "--experiment-root" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve()
        if value.startswith("--experiment-root="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def verify_iteration2_sources(root: Path) -> None:
    protocol = json.loads(
        (root / "protocols/evaluation_protocol.v1.json").read_text(encoding="utf-8")
    )
    bindings = protocol.get("iteration2", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-2 evaluation protocol lacks source bindings")
    for path_text, expected in bindings.items():
        path = REPO_ROOT / path_text
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Iteration-2 source binding mismatch: {path_text}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = requested_experiment_root()
    if root is None:
        raise SystemExit("Iteration-2 generation requires --experiment-root")
    verify_iteration2_sources(root)
    payload = load_module("iteration2_style_transfer_payloads", PAYLOAD_PATH)
    base = load_module("iteration2_generation_base", BASE_PATH)
    base.build_method_request = payload.build_method_request
    # Base provenance resolves the active runner and payload builder from __file__.
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
