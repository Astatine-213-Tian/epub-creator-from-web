from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from src.crawl.snapshot import write_json
from src.core.output import repo_root


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in Codex output")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'("zh"\s*:\s*"[^"]*?)(?=\}\s*(?:,\s*\{"index"|\]\s*,\s*"glossary_candidates"))',
            r'\1"',
            candidate,
        )
        if repaired != candidate:
            return json.loads(repaired)
        raise


def run_prompt_with_codex(
    *,
    prompt_path: Path,
    raw_output_path: Path,
    json_output_path: Path,
    model: str | None = None,
    codex_bin: str = "codex",
) -> dict[str, Any]:
    prompt = prompt_path.read_text(encoding="utf-8")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codex_bin,
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(raw_output_path),
        "-C",
        str(repo_root()),
        "-",
    ]
    if model:
        cmd[2:2] = ["--model", model]
    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = raw_output_path.with_name(f"{raw_output_path.stem}.log")
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
        )
    data = extract_json_object(raw_output_path.read_text(encoding="utf-8"))
    write_json(json_output_path, data)
    return data


def run_missing_prompts(
    *,
    run_dir: Path,
    model: str | None = None,
    codex_bin: str = "codex",
    chunk_ids: set[str] | None = None,
    overwrite: bool = False,
) -> int:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    completed = 0
    for chunk in manifest.get("chunks", []):
        chunk_id = str(chunk["chunk_id"])
        if chunk_ids is not None and chunk_id not in chunk_ids:
            continue
        prompt_path = run_dir / chunk["prompt_path"]
        raw_output_path = run_dir / chunk["raw_output_path"]
        json_output_path = run_dir / chunk["json_output_path"]
        if json_output_path.exists() and not overwrite:
            continue
        run_prompt_with_codex(
            prompt_path=prompt_path,
            raw_output_path=raw_output_path,
            json_output_path=json_output_path,
            model=model,
            codex_bin=codex_bin,
        )
        completed += 1
        print(f"translated chunk {chunk_id}")
    return completed
