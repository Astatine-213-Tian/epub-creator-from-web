"""Enforce the dependency boundaries that keep collection and output independent."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def imports(path: Path) -> set[str]:
    result = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


class ArchitectureTests(unittest.TestCase):
    def test_production_modules_do_not_depend_on_cli_or_research(self):
        for path in SRC.rglob("*.py"):
            if path.parent == SRC / "cli":
                continue
            with self.subTest(module=path.relative_to(ROOT)):
                self.assertFalse(
                    {m for m in imports(path) if m.startswith(("src.cli", "research"))}
                )

    def test_collection_content_and_notion_have_no_epub_output_dependency(self):
        forbidden = {
            "crawler": (
                "src.epub",
                "src.notion",
                "src.dataset",
                "src.translation",
                "src.workflows",
            ),
            "content": ("src.epub", "src.notion", "src.crawler", "src.workflows"),
            "notion": ("src.epub", "src.crawler", "src.workflows", "src.translation"),
            "metadata": (
                "src.epub",
                "src.crawler",
                "src.notion",
                "src.content",
                "src.workflows",
            ),
        }
        for folder, prefixes in forbidden.items():
            for path in (SRC / folder).rglob("*.py"):
                with self.subTest(module=path.relative_to(ROOT)):
                    self.assertFalse(
                        {m for m in imports(path) if m.startswith(prefixes)}
                    )

    def test_presentation_never_runs_source_cleanup_or_remote_services(self):
        forbidden = (
            "src.content.normalization",
            "src.content.prepare",
            "src.content.html",
            "src.epub.maintenance",
            "src.epub.normalize",
            "src.epub.review",
            "src.metadata",
            "src.notion",
            "src.crawler",
            "src.translation",
        )
        for name in ("writer", "xhtml", "bilingual"):
            with self.subTest(renderer=name):
                self.assertFalse(
                    {
                        m
                        for m in imports(SRC / "epub" / f"{name}.py")
                        if m.startswith(forbidden)
                    }
                )
