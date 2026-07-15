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


def argument(name: str) -> str | None:
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def verify_sources(root: Path) -> None:
    protocol_path = root / "protocols/evaluation_protocol.v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bindings = protocol.get("iteration3", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-3 protocol lacks frozen source bindings")
    for path_text, expected in bindings.items():
        observed = hashlib.sha256((REPO_ROOT / path_text).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Iteration-3 source binding mismatch: {path_text}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root_text = argument("--experiment-root")
    if root_text is None:
        raise SystemExit("Iteration-3 generation requires --experiment-root")
    root = Path(root_text).expanduser().resolve()
    stage = argument("--stage")
    method_id = argument("--method-id")
    if stage == "style_transfer":
        verify_sources(root)
        if method_id == "candidate_rerank":
            raise SystemExit(
                "candidate_rerank is deterministic; run build_candidate_rerank.py"
            )
    payload = load_module("iteration3_style_transfer_payloads", PAYLOAD_PATH)
    base = load_module("iteration3_generation_base", BASE_PATH)
    base.build_method_request = payload.build_method_request
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
