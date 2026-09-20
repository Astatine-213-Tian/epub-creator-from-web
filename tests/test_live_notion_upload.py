from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests import live_notion_upload as live


class LiveUploadRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        self.manifest = self.root / "run.json"
        self.checkpoint = self.root / "generated/notion_cms_sources/test/import.json"
        self.reader = SimpleNamespace(rows=AsyncMock(return_value=[]))

        async def connect(action):
            return await action(self.reader, None)

        mocked = patch.object(live, "with_notion", connect)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_failed_import_retains_partial_checkpoint_and_cleanup_checks_known_ids(self):
        def fail(*args):
            live.save(self.checkpoint, {"work_id": "created-work", "extras": [{}]})
            raise ValueError("readback failed")

        with patch.object(live, "write_outputs", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "readback failed"):
                live.import_book(self.root)
        run = live.load(self.manifest)
        self.assertEqual(run["import_state"], str(self.checkpoint))
        self.reader.rows.return_value = [{"id": "created-work"}]
        with self.assertRaisesRegex(AssertionError, "Test works still active"):
            live.verify_cleanup(self.root)
        self.reader.rows.return_value = []
        live.verify_cleanup(self.root)
        self.assertTrue(live.load(self.manifest)["notion_cleanup_verified"])

    def test_cleanup_finds_author_when_create_response_was_lost(self):
        with patch.object(live, "write_outputs", side_effect=ValueError("lost response")):
            with self.assertRaisesRegex(ValueError, "lost response"):
                live.import_book(self.root)
        run = live.load(self.manifest)
        self.reader.rows.side_effect = [[], [{
            "id": "unknown-author", live.FIELDS["authors"]: live.source(run).author,
        }]]
        with self.assertRaisesRegex(AssertionError, "Test authors still active"):
            live.verify_cleanup(self.root)
        self.reader.rows.side_effect = None
        live.verify_cleanup(self.root)
        self.assertTrue(live.load(self.manifest)["notion_cleanup_verified"])

    def test_failed_readback_keeps_author_ids_for_cleanup(self):
        def upload(book, options):
            live.save(self.checkpoint, {
                "work_id": "created-work", "metadata": {"title": book.title},
            })
            return SimpleNamespace(notion_state=self.checkpoint)

        async def document(page_id):
            if page_id == "created-work":
                run = live.load(self.manifest)
                return {
                    live.FIELDS["title"]: live.source(run).title,
                    live.FIELDS["authors"]: ["00000000-0000-4000-8000-000000000001"],
                }, ""
            raise ValueError("author readback failed")

        self.reader.document = document
        with patch.object(live, "write_outputs", side_effect=upload):
            with self.assertRaisesRegex(ValueError, "author readback failed"):
                live.import_book(self.root)
        self.assertEqual(live.load(self.manifest)["author_ids"], [
            "00000000-0000-4000-8000-000000000001",
        ])
