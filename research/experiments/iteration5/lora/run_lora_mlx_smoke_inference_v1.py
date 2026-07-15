#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "lora_paired_reconstruction_v1/runs/qwen3_4b_mlx_smoke_v1"
)
DATA = ROOT / "mlx_data" / "test.jsonl"
ADAPTERS = ROOT / "adapters"
OUTPUT = ROOT / "inference_v1"
PREREGISTRATION = OUTPUT / "preregistration.json"
MODEL = "Qwen/Qwen3-4B-MLX-4bit"
SEED = 20260718
MAX_TOKENS = 4096
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>|某某")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen base-versus-LoRA internal-test inference smoke."
    )
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--arm", choices=("base", "lora"), default="base")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample_id(row: dict[str, Any]) -> str:
    payload = json.loads(row["messages"][1]["content"])
    return str(payload["sample_id"])


def expected_contract(row: dict[str, Any]) -> tuple[str, list[str], dict[str, str]]:
    payload = json.loads(row["messages"][1]["content"])
    neutral = {str(item["id"]): str(item["zh"]) for item in payload["neutral_chinese"]}
    ids = [str(item["id"]) for item in payload["output_contract"]["paragraphs"]]
    return str(payload["sample_id"]), ids, neutral


def lock_payload() -> dict[str, Any]:
    rows = read_jsonl(DATA)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "qwen3-4b-mlx-lora-smoke-inference-v1",
        "status": "locked_before_base_or_lora_internal_test_generation",
        "claim_scope": "infrastructure_and_directional_feasibility_only",
        "model": MODEL,
        "adapter_path": str(ADAPTERS),
        "seed": SEED,
        "sampling": "greedy_temperature_0",
        "max_tokens": MAX_TOKENS,
        "test_rows": len(rows),
        "sample_ids": [sample_id(row) for row in rows],
        "input_hashes": {
            str(DATA): sha256_file(DATA),
            str(ADAPTERS / "adapter_config.json"): sha256_file(
                ADAPTERS / "adapter_config.json"
            ),
            str(ADAPTERS / "adapters.safetensors"): sha256_file(
                ADAPTERS / "adapters.safetensors"
            ),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "required_arms": ["base", "lora"],
        "validation": [
            "parse one JSON object",
            "sample_id equals the frozen test row",
            "paragraph IDs and order equal the frozen output contract",
            "every zh value is nonempty",
            "protected placeholder sequence equals the neutral input paragraph",
        ],
        "forbidden_claims": [
            "style-transfer efficacy",
            "superiority to prompt baselines",
            "production readiness",
            "generalization beyond the three internal-test books",
        ],
    }
    payload["lock_id"] = sha256_bytes(canonical(payload).encode("utf-8"))
    return payload


def validate_lock() -> dict[str, Any]:
    lock = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    expected = lock_payload()
    if lock != expected:
        raise ValueError("inference preregistration no longer matches frozen inputs")
    return lock


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.glob("base/*.json")):
        raise FileExistsError("base inference outputs already exist")
    if OUTPUT.exists() and any(OUTPUT.glob("lora/*.json")):
        raise FileExistsError("LoRA inference outputs already exist")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = lock_payload()
    if PREREGISTRATION.exists():
        if json.loads(PREREGISTRATION.read_text(encoding="utf-8")) != payload:
            raise FileExistsError("a different inference preregistration already exists")
        print(f"validated {PREREGISTRATION}")
        return
    PREREGISTRATION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {PREREGISTRATION}")
    print(f"lock_id={payload['lock_id']}")


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generated response is not a JSON object")
    return value


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def validate_result(
    result: dict[str, Any], row: dict[str, Any]
) -> tuple[bool, list[str]]:
    expected_sample_id, expected_ids, neutral = expected_contract(row)
    errors: list[str] = []
    if result.get("sample_id") != expected_sample_id:
        errors.append("sample_id mismatch")
    paragraphs = result.get("paragraphs")
    if not isinstance(paragraphs, list):
        return False, errors + ["paragraphs is not a list"]
    actual_ids = [str(item.get("id", "")) for item in paragraphs if isinstance(item, dict)]
    if actual_ids != expected_ids:
        errors.append("paragraph IDs or order mismatch")
    if len(paragraphs) != len(expected_ids):
        errors.append("paragraph count mismatch")
    for item in paragraphs:
        if not isinstance(item, dict):
            errors.append("paragraph is not an object")
            continue
        paragraph_id = str(item.get("id", ""))
        zh = item.get("zh")
        if not isinstance(zh, str) or not zh.strip():
            errors.append(f"{paragraph_id}: empty zh")
            continue
        if paragraph_id in neutral and placeholders(zh) != placeholders(neutral[paragraph_id]):
            errors.append(f"{paragraph_id}: protected placeholder sequence mismatch")
    return not errors, errors


def run(arm: str, resume: bool) -> None:
    lock = validate_lock()
    rows = read_jsonl(DATA)
    destination = OUTPUT / arm
    destination.mkdir(parents=True, exist_ok=True)

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    adapter_path = str(ADAPTERS) if arm == "lora" else None
    print(f"loading arm={arm} model={MODEL}", flush=True)
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)

    for index, row in enumerate(rows, start=1):
        current_sample_id = sample_id(row)
        output_path = destination / f"{current_sample_id}.json"
        if output_path.exists():
            if not resume:
                raise FileExistsError(f"output already exists: {output_path}")
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            if artifact.get("lock_id") != lock["lock_id"] or artifact.get("arm") != arm:
                raise ValueError(f"invalid resumed artifact: {output_path}")
            print(f"[{index}/{len(rows)}] skip {arm} {current_sample_id}", flush=True)
            continue

        messages = row["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        started = time.monotonic()
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.monotonic() - started
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        try:
            parsed = extract_json(raw)
            valid, errors = validate_result(parsed, row)
        except Exception as exc:  # The raw output is retained for audit.
            valid = False
            errors = [f"parse_error: {type(exc).__name__}: {exc}"]
        artifact = {
            "schema_version": 1,
            "experiment_id": lock["experiment_id"],
            "lock_id": lock["lock_id"],
            "arm": arm,
            "sample_id": current_sample_id,
            "model": MODEL,
            "adapter_path": str(ADAPTERS) if arm == "lora" else None,
            "seed": SEED,
            "elapsed_seconds": round(elapsed, 3),
            "valid": valid,
            "validation_errors": errors,
            "raw_output": raw,
            "result": parsed,
        }
        artifact["artifact_id"] = sha256_bytes(canonical(artifact).encode("utf-8"))
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{index}/{len(rows)}] wrote {arm} {current_sample_id} valid={valid} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    del model
    del tokenizer
    gc.collect()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run(args.arm, args.resume)


if __name__ == "__main__":
    main()
