from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.translate import (
    _validate_with_optional_semantic_qa,
    main as translate_main,
)
from src.crawl.snapshot import write_json
from src.runtime.progress import ProgressLogger
from src.translation.codex_cli import (
    ModelCapacityError,
    run_missing_prompts_with_model_fallback,
)
from src.translation.semantic_compression import (
    _qa_config_sha256,
    _review_contract,
    _sha256_json,
    _style_run_binding_sha256,
    run_semantic_compression_qa,
    semantic_qa_summary_is_current,
    translation_state_sha256,
)
from src.translation.style_transfer import (
    DEFAULT_MODEL_ORDER,
    RUN_SCHEMA,
    StyleTransferRunner,
    configured_model_order,
    validate_style_transfer_provenance,
)
from src.translation.style_transfer_assets import (
    METHOD_ID,
    build_method_request,
    file_sha256,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = (
    REPO_ROOT
    / "generated/author_styles/feitianyexiang/content_plan_combined.v1.json"
)
ASSET_SHA256 = "bf3ad5a97aafcb14cdcc6f11e12debdad9665883d7e66e78444ab839e6918645"
PROMPT_PATH = (
    REPO_ROOT
    / "book_specs/author_styles/feitianyexiang/content_plan_combined.prompt.md"
)
SCHEMA_PATH = (
    REPO_ROOT
    / "book_specs/author_styles/feitianyexiang/content_plan_combined.schema.json"
)


def successful_result(request: dict[str, object]) -> dict[str, object]:
    neutral = list(request["neutral_zh"])
    return {
        "sample_id": request["sample_id"],
        "method_id": request["method_id"],
        "intensity": request["intensity"],
        "content_plan": [
            {"id": row["id"], "facts": ["事件不变"], "constraints": ["语义不变"]}
            for row in neutral
        ],
        "paragraphs": [dict(row) for row in neutral],
        "style_cues_applied": [],
        "style_cues_skipped": [],
        "uncertainties": [],
    }


class ProductionStyleTransferTests(unittest.TestCase):
    def test_style_validation_requires_auto_review_when_not_skipped(self) -> None:
        with (
            patch("src.cli.translate.is_author_style_transfer_run", return_value=True),
            patch(
                "src.cli.translate.validate_style_transfer_provenance",
                return_value={"validated_chunk_count": 1},
            ),
            patch(
                "src.cli.translate.validate_translation_run",
                return_value={"chunks": [{}]},
            ),
        ):
            with self.assertRaises(ValueError):
                _validate_with_optional_semantic_qa(
                    run_dir=Path("/tmp/style-run"),
                    config={"semantic_compression": {"auto_review": False}},
                    allow_missing=False,
                    models=None,
                    codex_bin="codex",
                    timeout_seconds=None,
                    skip_semantic_qa=False,
                    semantic_overwrite=False,
                    progress=ProgressLogger(),
                )

    def test_all_command_always_runs_semantic_qa(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            neutral_run = root / "neutral"
            style_run = root / "style"
            output_path = root / "book.epub"
            with (
                patch(
                    "src.cli.translate.prepare_translation_run",
                    return_value=neutral_run,
                ),
                patch(
                    "src.cli.translate.validate_translation_run",
                    return_value={"glossary_candidate_count": 0},
                ),
                patch(
                    "src.cli.translate.prepare_author_style_transfer_run",
                    return_value=style_run,
                ),
                patch(
                    "src.cli.translate._validate_with_optional_semantic_qa"
                ) as validate_style,
                patch(
                    "src.cli.translate.build_epub_from_run",
                    return_value=output_path,
                ),
            ):
                return_code = translate_main(
                    ["all", str(root / "snapshot"), "--config", str(config_path)]
                )
            self.assertEqual(return_code, 0)
            self.assertFalse(validate_style.call_args.kwargs["skip_semantic_qa"])

    def test_default_model_order_prefers_all_local_56_variants(self) -> None:
        self.assertEqual(
            configured_model_order({}),
            [
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-5.6-luna",
                "gpt-5.5",
                "gpt-5.3-codex-spark",
                "gpt-5.4",
            ],
        )

    def test_asset_request_matches_frozen_method_evidence(self) -> None:
        old_path = (
            REPO_ROOT
            / "generated/translation_runs/eternal_gate_method4_20260714/"
            "content_plan_combined/requests/01_m4_001.json"
        )
        if not old_path.exists():
            self.skipTest("frozen method4 request is not present")
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = build_method_request(
            asset_path=ASSET_PATH,
            asset_file_sha256=ASSET_SHA256,
            sample_id=old["sample_id"],
            english_semantic_source=old["english_semantic_source"],
            neutral_zh=old["neutral_zh"],
        )
        self.assertEqual(new["reference_examples"], old["reference_examples"])
        self.assertEqual(
            new["method_payload"]["aligned_pair_retrieval"],
            old["method_payload"]["aligned_pair_retrieval"],
        )
        self.assertEqual(
            new["method_payload"]["generation_contract"],
            old["method_payload"]["generation_contract"],
        )
        self.assertEqual(
            new["method_payload"]["style_definition"],
            old["method_payload"]["style_definition"],
        )

    def test_style_runner_switches_only_after_capacity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = StyleTransferRunner(
                style_run_dir=root,
                asset_path=ASSET_PATH,
                asset_file_sha256=ASSET_SHA256,
                prompt_path=PROMPT_PATH,
                schema_path=SCHEMA_PATH,
                models=("gpt-5.6-sol", "gpt-5.6-terra"),
                reasoning_effort="high",
                timeout_seconds=30,
                max_attempts=2,
                codex_bin="codex",
                overwrite=False,
            )
            calls: list[str] = []

            def fake_attempt(**kwargs: object) -> dict[str, object]:
                model = str(kwargs["model"])
                calls.append(model)
                if model == "gpt-5.6-sol":
                    return {
                        "status": "failed",
                        "model": model,
                        "validation_errors": ["codex exited with status 1"],
                        "process_error_tail": "usage limit reached",
                        "stderr_tail": "",
                        "response_errors": [],
                        "response_usage": {},
                    }
                request = kwargs["request"]
                output_path = Path(kwargs["output_path"])
                result = successful_result(request)
                write_json(output_path, {"result": result})
                return {
                    "status": "success",
                    "model": model,
                    "response_id": "test-response",
                    "validation_errors": [],
                    "process_error_tail": "",
                    "stderr_tail": "",
                    "response_errors": [],
                    "response_usage": {},
                }

            with patch(
                "src.translation.style_transfer.run_structured_attempt",
                side_effect=fake_attempt,
            ):
                result = runner.invoke_segment(
                    chunk_id="01_style_001",
                    chapter_id="01",
                    rows=[
                        {
                            "index": 0,
                            "english": "He opened the gate.",
                            "neutral_zh": "他打开了门。",
                        }
                    ],
                    segment_name="root",
                    depth=0,
                )
            self.assertEqual(result.models_used, ["gpt-5.6-terra"])
            self.assertEqual(calls, ["gpt-5.6-sol", "gpt-5.6-terra"])
            self.assertIn("gpt-5.6-sol", runner.disabled_models())

    def test_style_runner_does_not_downgrade_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner = StyleTransferRunner(
                style_run_dir=Path(temp),
                asset_path=ASSET_PATH,
                asset_file_sha256=ASSET_SHA256,
                prompt_path=PROMPT_PATH,
                schema_path=SCHEMA_PATH,
                models=("gpt-5.6-sol", "gpt-5.6-terra"),
                reasoning_effort="high",
                timeout_seconds=30,
                max_attempts=2,
                codex_bin="codex",
                overwrite=False,
            )
            calls: list[str] = []

            def failed_attempt(**kwargs: object) -> dict[str, object]:
                model = str(kwargs["model"])
                calls.append(model)
                return {
                    "status": "failed",
                    "model": model,
                    "validation_errors": ["paragraph IDs or order do not match input"],
                    "process_error_tail": "",
                    "stderr_tail": "",
                    "response_errors": [],
                    "response_usage": {},
                }

            with patch(
                "src.translation.style_transfer.run_structured_attempt",
                side_effect=failed_attempt,
            ):
                with self.assertRaises(RuntimeError):
                    runner.invoke_segment(
                        chunk_id="01_style_001",
                        chapter_id="01",
                        rows=[
                            {
                                "index": 0,
                                "english": "He opened the gate.",
                                "neutral_zh": "他打开了门。",
                            }
                        ],
                        segment_name="root",
                        depth=0,
                    )
            self.assertEqual(calls, ["gpt-5.6-sol", "gpt-5.6-sol"])
            self.assertNotIn("gpt-5.6-terra", calls)

    def test_neutral_runner_resumes_on_next_model_after_capacity(self) -> None:
        base_summary = {
            "completed": 1,
            "skipped": 0,
            "full_success": 1,
            "chunk_failures": 0,
            "fallback_piece_attempts": 0,
            "fallback_piece_failures": 0,
            "local_window_attempts": 0,
            "local_window_failures": 0,
            "local_window_recovered": 0,
            "paragraph_attempts": 0,
            "paragraph_failures": 0,
            "recovered": 0,
            "still_missing": 0,
            "items": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            calls: list[str] = []

            def fake_run(**kwargs: object) -> dict[str, object]:
                model = str(kwargs["model"])
                calls.append(model)
                if model == "gpt-5.6-sol":
                    raise ModelCapacityError(model, "capacity exhausted")
                return dict(base_summary)

            with patch(
                "src.translation.codex_cli.run_missing_prompts",
                side_effect=fake_run,
            ):
                summary = run_missing_prompts_with_model_fallback(
                    run_dir=Path(temp),
                    models=DEFAULT_MODEL_ORDER[:2],
                )
            self.assertEqual(calls, ["gpt-5.6-sol", "gpt-5.6-terra"])
            self.assertEqual(summary["effective_model"], "gpt-5.6-terra")
            self.assertIn("gpt-5.6-sol", summary["unavailable_models"])

    def test_semantic_qa_summary_is_bound_to_style_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(
                root / "run_manifest.json",
                {
                    "chunks": [
                        {
                            "chunk_id": "01_style_001",
                            "json_output_path": "outputs/01_style_001.json",
                        }
                    ]
                },
            )
            output_path = root / "outputs/01_style_001.json"
            write_json(output_path, {"translations": [{"index": 0, "zh": "初稿"}]})
            summary_path = root / "semantic_compression/semantic_compression_summary.json"
            candidate_path = root / "semantic_compression/candidates.json"
            prompt_path = root / "semantic_compression/review_prompt.txt"
            result_path = root / "semantic_compression/review_result.json"
            write_json(candidate_path, {"candidates": []})
            prompt_path.write_text("review prompt\n", encoding="utf-8")
            write_json(result_path, {"items": []})
            write_json(
                summary_path,
                {
                    "output_translation_state_sha256": translation_state_sha256(root),
                    "dry_run": False,
                    "result_summary": {
                        "failure_count": 0,
                        "rejected_count": 0,
                        "unresolved_true_loss_count": 0,
                    },
                    "settings": {
                        "max_candidates": 30,
                        "batch_size": 8,
                        "auto_repair": False,
                        "min_auto_apply_confidence": "medium",
                    },
                    "provenance": {
                        "schema": "semantic_qa_provenance.v1",
                        "qa_config_sha256": _qa_config_sha256({}),
                        "review_contract_sha256": _sha256_json(_review_contract()),
                        "style_run_binding_sha256": _style_run_binding_sha256(root),
                        "style_source_state_sha256": "source-state",
                        "candidate_report_path": str(candidate_path),
                        "candidate_report_sha256": file_sha256(candidate_path),
                        "review_artifact_count": 1,
                        "review_artifacts": [
                            {
                                "prompt_path": str(prompt_path),
                                "prompt_sha256": file_sha256(prompt_path),
                                "result_path": str(result_path),
                                "result_sha256": file_sha256(result_path),
                            }
                        ],
                    },
                },
            )
            with patch(
                "src.translation.semantic_compression.validate_style_transfer_source_inputs",
                return_value={"state_sha256": "source-state"},
            ):
                self.assertTrue(
                    semantic_qa_summary_is_current(root, summary_path, {})
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["dry_run"] = True
                write_json(summary_path, summary)
                self.assertFalse(
                    semantic_qa_summary_is_current(root, summary_path, {})
                )
                summary["dry_run"] = False
                summary["result_summary"]["unresolved_true_loss_count"] = 1
                write_json(summary_path, summary)
                self.assertFalse(
                    semantic_qa_summary_is_current(root, summary_path, {})
                )
                summary["result_summary"]["unresolved_true_loss_count"] = 0
                write_json(summary_path, summary)
                write_json(
                    output_path, {"translations": [{"index": 0, "zh": "改稿"}]}
                )
                self.assertFalse(
                    semantic_qa_summary_is_current(root, summary_path, {})
                )

    def test_style_validation_rejects_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_run = root / "semantic"
            snapshot = root / "snapshot"
            write_json(semantic_run / "run_manifest.json", {"chunks": []})
            write_json(
                semantic_run / "translations/01.json",
                {"translations": [{"index": 0, "zh": "中性译文"}]},
            )
            write_json(
                snapshot / "manifest.json",
                {
                    "schema_version": 1,
                    "chapters": [{"id": "01", "path": "chapters/01.json"}],
                },
            )
            write_json(
                snapshot / "chapters/01.json",
                {"paragraphs": [{"index": 0, "english": "English source."}]},
            )
            request = {"sample_id": "s_000000000000000000000000"}
            request_sha256 = sha256_json(request)
            request_path = root / "requests/01_style_001.json"
            method_path = root / "method_outputs/01_style_001.json"
            output_path = root / "outputs/01_style_001.json"
            write_json(request_path, request)
            write_json(
                method_path,
                {
                    "run_schema": RUN_SCHEMA,
                    "method_id": METHOD_ID,
                    "request_sha256": request_sha256,
                },
            )
            write_json(
                output_path,
                {
                    "translations": [{"index": 0, "zh": "译文"}],
                    "style_transfer_provenance": {
                        "run_schema": RUN_SCHEMA,
                        "method_id": METHOD_ID,
                        "request_sha256": request_sha256,
                        "method_output_sha256": file_sha256(method_path),
                    },
                },
            )
            write_json(
                root / "run_manifest.json",
                {
                    "pass": "author_style_transfer",
                    "semantic_run_dir": str(semantic_run),
                    "snapshot_dir": str(snapshot),
                    "style_transfer": {
                        "schema": RUN_SCHEMA,
                        "asset_path": str(ASSET_PATH),
                        "asset_file_sha256": file_sha256(ASSET_PATH),
                        "prompt_path": str(PROMPT_PATH),
                        "prompt_sha256": file_sha256(PROMPT_PATH),
                        "schema_path": str(SCHEMA_PATH),
                        "schema_sha256": file_sha256(SCHEMA_PATH),
                        "semantic_manifest_sha256": file_sha256(
                            semantic_run / "run_manifest.json"
                        ),
                        "snapshot_manifest_sha256": file_sha256(
                            snapshot / "manifest.json"
                        ),
                    },
                    "chunks": [
                        {
                            "chunk_id": "01_style_001",
                            "chapter_id": "01",
                            "chapter_title": "Chapter 1",
                            "indexes": [0],
                            "request_path": "requests/01_style_001.json",
                            "method_output_path": "method_outputs/01_style_001.json",
                            "raw_output_path": "outputs/01_style_001.raw.txt",
                            "json_output_path": "outputs/01_style_001.json",
                            "request_sha256": request_sha256,
                            "english_sha256": sha256_json(["English source."]),
                            "neutral_zh_sha256": sha256_json(["中性译文"]),
                        }
                    ],
                },
            )
            self.assertEqual(
                validate_style_transfer_provenance(root)["validated_chunk_count"],
                1,
            )
            qa_config = {
                "semantic_compression": {
                    "auto_review": True,
                    "auto_repair": True,
                    "max_candidates": 30,
                    "batch_size": 8,
                    "min_auto_apply_confidence": "medium",
                }
            }
            qa_summary = run_semantic_compression_qa(
                run_dir=root,
                config=qa_config,
                max_candidates=30,
                batch_size=8,
                models=["gpt-5.6-sol"],
                auto_repair=True,
            )
            self.assertEqual(
                qa_summary["result_summary"]["unresolved_true_loss_count"], 0
            )
            self.assertTrue(
                semantic_qa_summary_is_current(
                    root,
                    root
                    / "semantic_compression/semantic_compression_summary.json",
                    qa_config,
                )
            )
            write_json(request_path, {"sample_id": "s_changed"})
            with self.assertRaises(ValueError):
                validate_style_transfer_provenance(root)
            write_json(request_path, request)
            write_json(
                semantic_run / "translations/01.json",
                {"translations": [{"index": 0, "zh": "改过的中性译文"}]},
            )
            with self.assertRaises(ValueError):
                validate_style_transfer_provenance(root)


if __name__ == "__main__":
    unittest.main()
