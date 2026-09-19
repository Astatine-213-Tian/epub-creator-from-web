from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.runtime.paths import repo_root
from src.metadata.jjwxc import GENRES, TIME_AREAS, JjwxcTypeMetadata


def classify_with_codex(
    *,
    title: str,
    author: str,
    sample: str,
    timeout_seconds: int = 240,
) -> JjwxcTypeMetadata | None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "time_area": {"type": "string", "enum": list(TIME_AREAS)},
            "genre": {"type": "string", "enum": list(GENRES)},
            "reason": {"type": "string"},
        },
        "required": ["time_area", "genre", "reason"],
    }
    prompt = f"""Classify this Chinese web novel for a local author-style dataset.

Choose exactly one time_area and one genre from the allowed labels.

Allowed time_area labels:
{", ".join(TIME_AREAS)}

Allowed genre labels:
{", ".join(GENRES)}

Use the article content signals, not the relationship orientation. If the evidence is weak, choose the most reasonable label.

Title: {title}
Author: {author}

Sample:
{sample[:6000]}
"""
    with tempfile.TemporaryDirectory(prefix="booklib-codex-classify-") as tmp_dir:
        tmp = Path(tmp_dir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "result.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ],
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root(),
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            return None
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    time_area = data.get("time_area", "")
    genre = data.get("genre", "")
    if time_area not in TIME_AREAS or genre not in GENRES:
        return None
    article_type = f"codex-inferred: {data.get('reason', '').strip()}"
    return JjwxcTypeMetadata(article_type, time_area, genre, "codex")


def classify_many_with_codex(
    items: list[dict[str, str]],
    *,
    timeout_seconds: int = 360,
) -> dict[str, JjwxcTypeMetadata]:
    if not items:
        return {}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "key": {"type": "string"},
                        "time_area": {"type": "string", "enum": list(TIME_AREAS)},
                        "genre": {"type": "string", "enum": list(GENRES)},
                        "reason": {"type": "string"},
                    },
                    "required": ["key", "time_area", "genre", "reason"],
                },
            }
        },
        "required": ["items"],
    }
    payload = [
        {
            "key": item["key"],
            "title": item["title"],
            "author": item["author"],
            "sample": item.get("sample", "")[:1600],
        }
        for item in items
    ]
    prompt = f"""Classify Chinese web novels for a local author-style dataset.

For each item, choose exactly one time_area and one genre from the allowed labels.

Allowed time_area labels:
{", ".join(TIME_AREAS)}

Allowed genre labels:
{", ".join(GENRES)}

Use article content signals, setting, plot, and title. Do not classify by relationship orientation. If evidence is weak, choose the most reasonable label.

Return one result for every input key.

INPUT JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    with tempfile.TemporaryDirectory(prefix="booklib-codex-classify-batch-") as tmp_dir:
        tmp = Path(tmp_dir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "result.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ],
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root(),
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            return {}
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    output: dict[str, JjwxcTypeMetadata] = {}
    for item in data.get("items", []):
        key = item.get("key", "")
        time_area = item.get("time_area", "")
        genre = item.get("genre", "")
        if not key or time_area not in TIME_AREAS or genre not in GENRES:
            continue
        article_type = f"codex-inferred: {item.get('reason', '').strip()}"
        output[key] = JjwxcTypeMetadata(article_type, time_area, genre, "codex")
    return output





def metadata_to_cache(metadata: JjwxcTypeMetadata) -> dict[str, str]:
    return asdict(metadata)
