#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.crawler.snapshot import write_json
from src.translation.style_transfer_assets import (
    ASSET_SCHEMA,
    METHOD_ID,
    file_sha256,
    sha256_json,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = RESEARCH_ROOT.parent
DEFAULT_SOURCE_ASSET = (
    RESEARCH_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/method_assets/style_transfer_payloads.v1/"
    "assets.497a0db8919ccc9cfd97f6a729ff0528953d2d97477e83e07e55c42e4bb994d8.json"
)
DEFAULT_SOURCE_FILE_SHA256 = (
    "469a4f91c8871bd6c88b5fea130ed860d0d717a27c06ef10240bbb17b43a511f"
)
DEFAULT_OUTPUT = (
    PRODUCTION_ROOT
    / "generated/author_styles/feitianyexiang/content_plan_combined.v1.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_asset(
    asset_path: Path, *, expected_file_sha256: str
) -> dict[str, Any]:
    if file_sha256(asset_path) != expected_file_sha256:
        raise ValueError(f"frozen research asset file hash mismatch: {asset_path}")
    bundle = read_json(asset_path)
    if sha256_json(bundle.get("assets")) != bundle.get("content_sha256"):
        raise ValueError(f"frozen research asset content hash mismatch: {asset_path}")
    return bundle


def build_production_bundle(source_path: Path, source_bundle: dict[str, Any]) -> dict[str, Any]:
    source_assets = source_bundle["assets"]
    method = source_assets["methods"][METHOD_ID]["intensities"]["strong"]
    assets = {
        "target_author": source_assets["target_author"],
        "method": method,
        "aligned_pairs": source_assets["aligned_pairs"],
        "style_definition": source_assets["style_definition"],
    }
    assets["component_hashes"] = {
        "aligned_pairs_sha256": sha256_json(assets["aligned_pairs"]),
        "style_definition_sha256": sha256_json(assets["style_definition"]),
        "method_sha256": sha256_json(assets["method"]),
    }
    return {
        "schema_version": 1,
        "asset_schema": ASSET_SCHEMA,
        "content_sha256": sha256_json(assets),
        "source": {
            "research_asset_path": str(source_path.relative_to(RESEARCH_ROOT)),
            "research_asset_file_sha256": file_sha256(source_path),
            "research_asset_content_sha256": source_bundle["content_sha256"],
            "method_id": METHOD_ID,
            "intensity": "strong",
        },
        "assets": assets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen content-plan style method as a production asset."
    )
    parser.add_argument("--source-asset", type=Path, default=DEFAULT_SOURCE_ASSET)
    parser.add_argument(
        "--source-file-sha256", default=DEFAULT_SOURCE_FILE_SHA256
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source_asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_bundle = load_source_asset(
        source_path, expected_file_sha256=args.source_file_sha256
    )
    bundle = build_production_bundle(source_path, source_bundle)
    if output.exists() and not args.overwrite:
        existing = read_json(output)
        if existing != bundle:
            raise FileExistsError(f"production asset already exists with different content: {output}")
    else:
        write_json(output, bundle)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": file_sha256(output),
                "content_sha256": bundle["content_sha256"],
                "aligned_pair_count": bundle["assets"]["aligned_pairs"]["count"],
                "masked_example_count": len(
                    bundle["assets"]["style_definition"]["definition"][
                        "masked_scene_examples"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
