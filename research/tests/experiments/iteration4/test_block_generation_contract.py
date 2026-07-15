#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from experiments.iteration4 import run_style_transfer_block_generation as block_runner

from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_result(
    stage: str, request: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    if stage != "style_transfer":
        return []
    expected = [row["id"] for row in request["neutral_zh"]]
    observed = [row.get("id") for row in result.get("paragraphs", [])]
    return [] if observed == expected else ["paragraph IDs or order do not match input"]


def fake_base(failure_rule: Callable[[list[dict[str, Any]]], bool]) -> SimpleNamespace:
    def original(**kwargs: Any) -> dict[str, Any]:
        request = kwargs["request"]
        neutral = request["neutral_zh"]
        output_path = kwargs["output_path"]
        input_sha = sha256_text(canonical_json(request))
        if failure_rule(neutral):
            return {
                "started_at": "2026-07-13T00:00:00+00:00",
                "completed_at": "2026-07-13T00:00:01+00:00",
                "status": "failed",
                "input_sha256": input_sha,
                "output_sha256": None,
                "response_id": "failed-response",
                "response_usage": {"input_tokens": 10, "output_tokens": 2},
                "response_file": str(output_path.relative_to(REPO_ROOT)),
                "validation_errors": ["paragraph IDs or order do not match input"],
            }
        result = {
            "sample_id": request["sample_id"],
            "method_id": request["method_id"],
            "intensity": request["intensity"],
            "content_plan": (
                [
                    {
                        "id": row["id"],
                        "facts": [f"fact-{index}"],
                        "constraints": [f"constraint-{index}"],
                    }
                    for index, row in enumerate(neutral)
                ]
                if request["method_id"]
                == "content_plan_combined_full_regeneration"
                else []
            ),
            "paragraphs": [
                {"id": row["id"], "zh": row["zh"] + "-styled"} for row in neutral
            ],
            "style_cues_applied": ["test-cue"],
            "style_cues_skipped": [],
            "uncertainties": [],
        }
        output_sha = sha256_text(canonical_json(result))
        write_json(output_path, {"result": result})
        return {
            "started_at": "2026-07-13T00:00:00+00:00",
            "completed_at": "2026-07-13T00:00:01+00:00",
            "status": "success",
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "response_id": "success-response",
            "response_usage": {"input_tokens": 10, "output_tokens": 2},
            "response_file": str(output_path.relative_to(REPO_ROOT)),
            "validation_errors": [],
        }

    return SimpleNamespace(
        run_one_attempt=original,
        canonical_json=canonical_json,
        sha256_text=sha256_text,
        validate_result=validate_result,
        write_json=write_json,
    )


def request(
    paragraph_count: int,
    *,
    offset: int = 40,
    method_id: str = "aligned_pairs_full_regeneration",
) -> dict[str, Any]:
    ids = [f"p{offset + index + 1:04d}" for index in range(paragraph_count)]
    return {
        "sample_id": "s_0123456789abcdef01234567",
        "method_id": method_id,
        "intensity": "strong",
        "english_semantic_source": [
            {"id": paragraph_id, "en": f"English {index}"}
            for index, paragraph_id in enumerate(ids)
        ],
        "neutral_zh": [
            {"id": paragraph_id, "zh": f"neutral-{index}"}
            for index, paragraph_id in enumerate(ids)
        ],
        "method_payload": {},
        "reference_examples": [],
    }


def run_kwargs(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "stage": "style_transfer",
        "request": payload,
        "output_path": output_path,
        "sample_id": payload["sample_id"],
        "attempt": 1,
        "run_id": "iteration4_contract_test",
        "model_config": {
            "codex_model": "fake-model",
            "reasoning_effort": "high",
        },
        "prompt_sha256": "prompt-sha",
        "execution_binding": {},
    }


class BlockGenerationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        debug_root = REPO_ROOT / "generated/style_research/style_transfer_experiments/debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="block-contract-", dir=debug_root)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_short_segment_uses_local_ids_and_restores_source_ids(self) -> None:
        base = fake_base(lambda _rows: False)
        block_runner.install_block_generation(base)
        payload = request(3)
        output = self.root / "short.json"

        record = base.run_one_attempt(**run_kwargs(payload, output))

        self.assertEqual(record["status"], "success")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        observed = [row["id"] for row in artifact["result"]["paragraphs"]]
        expected = [row["id"] for row in payload["neutral_zh"]]
        self.assertEqual(observed, expected)
        segment = artifact["block_execution"]["segments"][0]
        self.assertEqual(segment["local_ids"], ["p0001", "p0002", "p0003"])
        self.assertEqual(segment["source_ids_sha256"], sha256_text(canonical_json(expected)))

    def test_failed_segment_is_bisected_and_reassembled_in_order(self) -> None:
        base = fake_base(lambda rows: len(rows) > 2)
        block_runner.install_block_generation(base)
        payload = request(4)
        output = self.root / "split.json"

        record = base.run_one_attempt(**run_kwargs(payload, output))

        self.assertEqual(record["status"], "success")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        expected = [row["id"] for row in payload["neutral_zh"]]
        observed = [row["id"] for row in artifact["result"]["paragraphs"]]
        self.assertEqual(observed, expected)
        execution = artifact["block_execution"]
        self.assertEqual(execution["attempted_segment_count"], 3)
        self.assertEqual(execution["leaf_segment_count"], 2)
        self.assertEqual(record["block_execution"]["recovered_split_count"], 1)
        self.assertEqual(record["response_usage"]["input_tokens"], 30)

    def test_content_plan_uses_source_ids_after_block_merge(self) -> None:
        base = fake_base(lambda _rows: False)
        block_runner.install_block_generation(base)
        payload = request(
            14,
            method_id="content_plan_combined_full_regeneration",
        )
        output = self.root / "content-plan.json"

        record = base.run_one_attempt(**run_kwargs(payload, output))

        self.assertEqual(record["status"], "success")
        artifact = json.loads(output.read_text(encoding="utf-8"))
        expected = [row["id"] for row in payload["neutral_zh"]]
        observed = [row["id"] for row in artifact["result"]["content_plan"]]
        self.assertEqual(observed, expected)
        self.assertEqual(
            artifact["block_execution"]["leaf_segment_count"],
            2,
        )
        self.assertTrue(
            all(row["facts"] and row["constraints"] for row in artifact["result"]["content_plan"])
        )

    def test_single_paragraph_failure_remains_a_failed_attempt(self) -> None:
        base = fake_base(lambda _rows: True)
        block_runner.install_block_generation(base)
        payload = request(1)
        output = self.root / "failed.json"

        record = base.run_one_attempt(**run_kwargs(payload, output))

        self.assertEqual(record["status"], "failed")
        self.assertFalse(output.exists())
        self.assertIn("block_generation_failed:s0001_e0001_d00", record["validation_errors"])


if __name__ == "__main__":
    unittest.main()
