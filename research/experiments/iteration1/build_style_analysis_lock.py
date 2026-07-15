#!/usr/bin/env python3
from __future__ import annotations

"""Create the content-addressed pre-style analysis lock."""

import argparse
import json
from pathlib import Path

from experiments.iteration1.style_analysis_lock import (
    DEFAULT_EXPERIMENT_ROOT,
    build_lock_payload,
    display_path,
    file_sha256,
    validate_analysis_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze all analysis code and artifacts before style generation."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    payload = build_lock_payload(root)
    output = args.output or (
        root
        / "protocols"
        / f"pre_style_analysis_lock.v1.{payload['content_sha256']}.json"
    )
    output = output.resolve()
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace a different analysis lock: {output}")
        action = "reused_identical"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        action = "created"
    binding = validate_analysis_lock(output, root)
    print(
        json.dumps(
            {
                "status": action,
                "output": display_path(output),
                "output_sha256": file_sha256(output),
                "content_sha256": payload["content_sha256"],
                "source_count": len(payload["source_bindings"]),
                "artifact_count": len(payload["artifact_bindings"]),
                "binding": binding,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
