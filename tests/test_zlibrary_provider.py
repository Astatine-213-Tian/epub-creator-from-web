from __future__ import annotations

import unittest

from src.crawler.providers.zlibrary import parser, search


DETAIL_HTML = """
<html>
  <head><meta name="description" content="完整的测试描述"></head>
  <body>
    <h1 class="book-title">我的一个朋友</h1>
    <i class="authors"><a>孔恰</a></i>
    <a class="addDownloadedBook" href="/dl/example">epub, 369 KB</a>
    <div class="bookProperty property_publisher">
      <div class="property_value">MoyuStudio</div>
    </div>
    <div class="bookProperty property_year"><div class="property_value">2018</div></div>
    <div class="bookProperty property_language"><div class="property_value">中文</div></div>
    <div class="bookProperty property_isbn"><div class="property_value">B07LBQGHSM</div></div>
  </body>
</html>
"""

SEARCH_HTML = """
<html><body>
  <z-bookcard href="/book/ignored/one.html" extension="azw3" publisher="MoyuStudio">
    <div slot="title">我的一个朋友</div><div slot="author">孔恰</div>
  </z-bookcard>
  <z-bookcard href="/book/plain/one.html" extension="epub" filesize="2 MB"
      publisher="嘉鱼出版社" year="2017" quality="5.0">
    <div slot="title">我的一个朋友</div><div slot="author">孔恰</div>
  </z-bookcard>
  <z-bookcard href="/book/preferred/one.html" extension="epub" filesize="369 KB"
      publisher="MoyuStudio" year="2018" quality="0.0">
    <div slot="title">我的一个朋友</div><div slot="author">孔恰</div>
  </z-bookcard>
  <div hidden>
    <z-bookcard href="/book/stale/one.html" extension="epub" publisher="MoyuStudio">
      <div slot="title">我的一个朋友</div><div slot="author">孔恰</div>
    </z-bookcard>
  </div>
</body></html>
"""


class ZLibraryProviderTests(unittest.TestCase):
    def test_parse_detail_page(self) -> None:
        meta = parser.parse_detail_page(
            DETAIL_HTML,
            "https://zh.1lib.sk/book/6ZD8jBKNZg/example.html",
        )

        self.assertEqual(meta.book_id, "6ZD8jBKNZg")
        self.assertEqual(meta.title, "我的一个朋友")
        self.assertEqual(meta.author, "孔恰")
        self.assertEqual(meta.download_url, "https://zh.1lib.sk/dl/example")
        self.assertEqual(meta.extension, "epub")
        self.assertEqual(meta.file_size, "369 KB")
        self.assertEqual(meta.publisher, "MoyuStudio")
        self.assertEqual(meta.isbn, "B07LBQGHSM")

    def test_search_keeps_epub_and_prefers_moyustudio(self) -> None:
        results = search.parse_search_page(SEARCH_HTML)

        self.assertEqual([result.url for result in results], [
            "https://zh.1lib.sk/book/preferred/one.html",
            "https://zh.1lib.sk/book/plain/one.html",
        ])
        self.assertTrue(all(result.author == "孔恰" for result in results))

    def test_resolve_book_id(self) -> None:
        self.assertEqual(
            parser._resolve_book_url("6ZD8jBKNZg"),
            "https://zh.1lib.sk/book/6ZD8jBKNZg",
        )


if __name__ == "__main__":
    unittest.main()
