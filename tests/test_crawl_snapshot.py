from __future__ import annotations

import unittest

from src.crawler.snapshot import deduplicate_comments


class CrawlSnapshotTests(unittest.TestCase):
    def test_comment_ids_are_deduplicated_before_snapshot_storage(self) -> None:
        comments = [
            {"id": "comment-1", "body": "first copy"},
            {"id": "comment-1", "body": "duplicate API copy"},
            {
                "id": "",
                "parent_id": "comment-1",
                "author": "Risk",
                "body": "中文回复",
                "created": "2026-01-01",
            },
            {
                "id": "",
                "parent_id": "comment-1",
                "author": "Risk",
                "body": "中文回复",
                "created": "2026-01-01",
            },
            {"id": "comment-2", "body": "another comment"},
        ]

        self.assertEqual(
            deduplicate_comments(comments),
            [comments[0], comments[2], comments[3], comments[4]],
        )


if __name__ == "__main__":
    unittest.main()
