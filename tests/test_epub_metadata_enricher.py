from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from lxml import etree

from src.metadata.epub_enricher import (
    DC_NS,
    OPF_NS,
    MetadataLookup,
    PackageMetadata,
    SourceMetadata,
    enrich_epub_metadata,
    parse_jjwxc_author_catalog,
    parse_jjwxc_author_search,
)


OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
    <dc:identifier id="id">fixture</dc:identifier>
    <dc:title>千秋</dc:title>
    <dc:creator id="creator">梦溪石</dc:creator>
    <dc:language>zh-CN</dc:language>
    <dc:date>1999-01-01</dc:date>
    <dc:source>https://example.invalid/old</dc:source>
    <dc:subject>旧标签</dc:subject>
  </metadata>
  <manifest/>
  <spine/>
</package>
"""

AUTHOR_SEARCH = """
<html><body>
  <a href="http://www.jjwxc.net/oneauthor.php?authorid=425111">
    <span><span>梦溪石</span></span>
  </a>
</body></html>
"""

AUTHOR_CATALOG = """
<html><body>
  <table class="author series">
    <tr><td colspan="3" style="font-weight: bold;">【耽美】</td></tr>
  </table>
  <table class="author novel">
    <tr><td>
      <a onclick="jump_book2(2423737,425111)">〖剑胆琴心〗《千秋》</a>
    </td></tr>
    <tr><td>类型:原创-纯爱-架空历史-传奇-主受</td></tr>
    <tr><td>发表时间：2015-10-27 20:00:00</td></tr>
  </table>
  <table class="author novel">
    <tr><td>
      <a onclick="jump_book2(3542625,425111)">〖剑胆琴心〗《无双》</a>
    </td></tr>
    <tr><td>类型:原创-纯爱-古色古香-传奇-主受</td></tr>
    <tr><td>发表时间：2018-04-30 15:47:00</td></tr>
  </table>
</body></html>
"""


HEADER_SERIES_CATALOG = """
<html><body>
  <table class="author series">
    <tr><td colspan="3" style="font-weight: bold;">【山海幻世】</td></tr>
  </table>
  <table class="author novel">
    <tr><td>
      <a onclick="jump_book2(7437872,293673)">〖龙吟〗《万物风华录》</a>
    </td></tr>
    <tr><td>类型:原创-纯爱-近代现代-爱情-主受</td></tr>
    <tr><td>发表时间：2022-01-01 00:00:00</td></tr>
  </table>
  <table class="author novel">
    <tr><td>
      <a onclick="jump_book2(3848563,293673)">〖心灯〗《定海浮生录》</a>
    </td></tr>
    <tr><td>类型:原创-纯爱-古色古香-传奇-主受</td></tr>
    <tr><td>发表时间：2018-01-01 00:00:00</td></tr>
  </table>
  <table class="author novel">
    <tr><td>
      <a onclick="jump_book2(3145450,293673)">〖真火〗《天宝伏妖录》</a>
    </td></tr>
    <tr><td>类型:原创-纯爱-古色古香-传奇-主受</td></tr>
    <tr><td>发表时间：2016-01-01 00:00:00</td></tr>
  </table>
  <table class="author series">
    <tr><td colspan="3" style="font-weight: bold;">【角落里的事】</td></tr>
  </table>
  <table class="author novel">
    <tr><td><a onclick="jump_book2(9999,293673)">〖同志〗《北城天街》</a></td></tr>
    <tr><td>发表时间：2015-01-01 00:00:00</td></tr>
  </table>
</body></html>
"""


class _StaticLookup(MetadataLookup):
    def __init__(self, source: SourceMetadata) -> None:
        self.source = source

    def find(
        self,
        package: PackageMetadata,
        *,
        primary_subject: str = "耽美",
    ) -> SourceMetadata | None:
        return self.source


class EpubMetadataEnricherTests(unittest.TestCase):
    def test_parses_exact_author_and_jjwxc_series_positions(self) -> None:
        self.assertEqual(
            parse_jjwxc_author_search(AUTHOR_SEARCH, "梦溪石"),
            "425111",
        )
        candidates = parse_jjwxc_author_catalog(AUTHOR_CATALOG)
        self.assertEqual([candidate.title for candidate in candidates], ["千秋", "无双"])
        self.assertEqual([candidate.series for candidate in candidates], ["剑胆琴心", "剑胆琴心"])
        self.assertEqual(
            [candidate.series_position for candidate in candidates],
            [1, 2],
        )
        self.assertEqual(candidates[0].date, "2015-10-27")
        self.assertEqual(
            candidates[0].raw_type,
            "原创-纯爱-架空历史-传奇-主受",
        )

    def test_uses_meaningful_catalog_section_for_unique_book_prefixes(self) -> None:
        candidates = parse_jjwxc_author_catalog(HEADER_SERIES_CATALOG)

        self.assertEqual(
            [candidate.series for candidate in candidates],
            ["山海幻世", "山海幻世", "山海幻世", None],
        )
        self.assertEqual(
            [candidate.series_position for candidate in candidates],
            [1, 2, 3, None],
        )

    def test_jinjiang_omits_aiqing_when_danmei_is_present(self) -> None:
        catalog = """
        <html><body>
          <table class="author novel">
            <tr><td><a onclick="jump_book2(2423737,425111)">《千秋》</a></td></tr>
            <tr><td>类型:原创-纯爱-近代现代-爱情-主受</td></tr>
            <tr><td>发表时间：2015-10-27 20:00:00 [锁]</td></tr>
          </table>
        </body></html>
        """
        package = PackageMetadata(
            title="千秋",
            author="梦溪石",
            language="zh-CN",
            date="",
            source="",
            description="",
            subjects=(),
            series=None,
            series_position=None,
        )
        lookup = MetadataLookup(
            author_ids={"梦溪石": "425111"},
            kadokado=False,
        )
        lookup._catalog_cache["425111"] = parse_jjwxc_author_catalog(catalog)

        danmei = lookup.find(package)
        self.assertIsNotNone(danmei)
        assert danmei is not None
        self.assertEqual(danmei.subjects, ("耽美", "近代现代"))

        non_danmei = lookup.find(package, primary_subject="言情")
        self.assertIsNotNone(non_danmei)
        assert non_danmei is not None
        self.assertEqual(non_danmei.subjects, ("言情", "爱情", "近代现代"))

    def test_jinjiang_date_overwrites_existing_date_and_is_idempotent(self) -> None:
        source = SourceMetadata(
            provider="jinjiang",
            title="千秋",
            author="梦溪石",
            language="zh-CN",
            date="2015-10-27",
            source="https://m.jjwxc.net/book2/2423737",
            description="一句话简介：千秋",
            subjects=("耽美", "传奇", "架空历史"),
            series="剑胆琴心",
            series_position=1,
            series_verified=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "千秋.epub"
            backup_root = root / "backups"
            self._write_epub(target)

            first = enrich_epub_metadata(
                target,
                backup_dir=backup_root,
                overwrite_backup=True,
                lookup=_StaticLookup(source),
            )
            self.assertEqual(first.status, "enriched")
            self.assertIn("date", first.changed_fields)
            self.assertEqual(first.publication_date, "2015-10-27")
            self.assertTrue((backup_root / target.name).exists())

            with zipfile.ZipFile(target) as archive:
                root_element = etree.fromstring(archive.read("EPUB/content.opf"))
            metadata = root_element.find(f"{{{OPF_NS}}}metadata")
            assert metadata is not None
            self.assertEqual(
                self._texts(metadata, "date"),
                ["2015-10-27"],
            )
            self.assertEqual(
                self._texts(metadata, "subject"),
                ["耽美", "传奇", "架空历史"],
            )
            collection = next(
                element
                for element in metadata.findall(f"{{{OPF_NS}}}meta")
                if element.get("property") == "belongs-to-collection"
            )
            self.assertEqual(collection.text, "剑胆琴心")

            before_second = target.read_bytes()
            second = enrich_epub_metadata(
                target,
                backup_dir=backup_root,
                overwrite_backup=True,
                lookup=_StaticLookup(source),
            )
            self.assertEqual(second.status, "already_complete")
            self.assertFalse(second.changed)
            self.assertEqual(target.read_bytes(), before_second)

    def test_check_mode_reports_without_rewriting(self) -> None:
        source = SourceMetadata(
            provider="jinjiang",
            title="千秋",
            author="梦溪石",
            language="zh-CN",
            date="2015-10-27",
            source="https://m.jjwxc.net/book2/2423737",
            description="",
            subjects=("耽美", "传奇", "架空历史"),
            series=None,
            series_verified=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "千秋.epub"
            self._write_epub(target)
            original = target.read_bytes()
            report = enrich_epub_metadata(
                target,
                apply=False,
                lookup=_StaticLookup(source),
            )
            self.assertTrue(report.changed)
            self.assertFalse(report.applied)
            self.assertEqual(target.read_bytes(), original)

    @staticmethod
    def _texts(metadata: etree._Element, local_name: str) -> list[str]:
        return [
            element.text or ""
            for element in metadata.findall(f"{{{DC_NS}}}{local_name}")
        ]

    @staticmethod
    def _write_epub(path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "mimetype",
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            archive.writestr("EPUB/content.opf", OPF)


if __name__ == "__main__":
    unittest.main()
