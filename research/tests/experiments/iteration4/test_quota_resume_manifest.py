from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.iteration4 import build_quota_resume_manifest as MODULE

class QuotaResumeManifestTest(unittest.TestCase):
    def test_builds_resume_state_without_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment"
            run_root = root / "runs" / "sample_set" / "run_id"
            output_dir = run_root / "english_source_repair_qa" / "round_01"
            ledger_dir = run_root / "ledgers"
            selection_path = root / "sample_sets" / "repair.json"
            output_dir.mkdir(parents=True)
            ledger_dir.mkdir(parents=True)
            selection_path.parent.mkdir(parents=True)
            selection_path.write_text(
                json.dumps({"sample_ids": ["s_done", "s_pending"]}),
                encoding="utf-8",
            )
            (output_dir / "s_done.json").write_text(
                json.dumps({"sample_id": "s_done", "result": {"approved": True}}),
                encoding="utf-8",
            )
            limit_message = (
                "You've hit your usage limit. Visit settings or try again at "
                "Jul 19th, 2026 3:16 PM."
            )
            rows = [
                {"sample_id": "s_done", "status": "success", "model": "gpt-5.4"},
                {
                    "sample_id": "s_pending",
                    "status": "failed",
                    "model": "gpt-5.4",
                    "completed_at": "2026-07-13T00:00:00Z",
                    "response_errors": [{"message": limit_message}],
                },
            ]
            ledger_path = ledger_dir / "english_source_repair_qa.round_01.jsonl"
            ledger_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            manifest_path = run_root / "resume.json"
            argv = [
                str(Path(MODULE.__file__)),
                "--experiment-root",
                str(root),
                "--sample-set",
                "sample_set",
                "--run-id",
                "run_id",
                "--stage",
                "english_source_repair_qa",
                "--selection-file",
                str(selection_path),
                "--repair-round",
                "1",
                "--jobs",
                "6",
                "--output",
                str(manifest_path),
            ]
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    MODULE.main()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["completed_sample_count"], 1)
            self.assertEqual(manifest["pending_sample_count"], 1)
            self.assertEqual(
                manifest["quota_reset_at_provider_text"], "Jul 19th, 2026 3:16 PM"
            )
            self.assertEqual(manifest["resume_argv"][-1], "--resume")
            self.assertNotIn("result", manifest)


if __name__ == "__main__":
    unittest.main()
