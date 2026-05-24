---
name: epub-reformatter
description: >
  Repair and normalize generated EPUB files in this project. Use when Codex needs
  to patch EPUB archives for Chinese novel formatting issues: volume/fanwai/houji
  TOC hierarchy, nav.xhtml and toc.ncx sync, spine ordering, duplicate intro or
  volume marker cleanup, chapter title normalization, author/book-name title
  cleanup, and validation of files under the project epub/ directory.
---

# EPUB Reformatter

## Core Rule

Patch EPUB archives directly and keep every reader-visible surface in sync:

- `EPUB/nav.xhtml`
- `EPUB/toc.ncx`
- affected chapter XHTML files
- `EPUB/content.opf` manifest/spine when adding, deleting, splitting, merging, or moving reading-order files

Always make a temporary backup before archive surgery, rewrite to a temporary archive, replace atomically, then validate. Delete backup files when the user asks for cleanup.

## Workflow

1. Inspect `nav.xhtml`, `toc.ncx`, `content.opf`, and the affected chapter XHTML.
2. Decide hierarchy from actual chapter titles and source markers, not only parser output.
3. Patch chapter files first, then rebuild `nav.xhtml` and `toc.ncx` from the intended hierarchy.
4. If reading order changes, reorder `content.opf` spine to match.
5. Validate:
   - `python3 -m zipfile -t epub/<book>.epub`
   - XML parse `EPUB/nav.xhtml`, `EPUB/toc.ncx`, `EPUB/content.opf`, and changed chapter files
   - `uv run python src/cli/validate_epub_chapters.py epub/<book>.epub`

## Hierarchy Rules

- `番外` must be a top-level group.
- If source content has `番外卷...`, delete that marker from the chapter body and start top-level `番外` at the next chapter. Do not use the `番外卷` name as the group title.
- Detect trailing `番外` from both TOC labels and chapter body content. Common signals:
  - everything after final-main markers such as `终章`, `尾声`, `后记`, or body text like `全文完` / `正文完`
  - chapter titles that start with `番外` or contain obvious extra labels such as `中秋番外`, `特典`, `番外一`
  - body-only markers such as `番外卷...`
- When `番外卷...` is found inside a chapter body, treat the next chapter as the start of `番外`, remove the marker text, and keep the group name simply `番外`.
- If the `番外` boundary is ambiguous after inspecting TOC labels and chapter body markers, stop and confirm the start chapter with the user instead of guessing.
- Merge consecutive multipart fanwai chapters when they share the same base title and differ only by part suffix, for example `第114章 沧浪之龙（一）`, `第115章 沧浪之龙（二）`, etc. The merged chapter title should drop the part suffix, each original part should be separated by a centered divider such as `（一）`, `（二）`, and all following numbered chapter titles must be renumbered incrementally across chapter XHTML, nav, and NCX.
- `后记` should be top-level and should appear before any `番外` group.
- `尾声` usually belongs to the last main volume. If the user says it should be top-level, place it before `后记` and before `番外`.
- `序言` / `序章` / prologue text is chapter-like content, not intro metadata. Treat it like chapter 0: split it into its own top-level chapter before `第1章`, add it to spine/nav/toc, and remove it from `intro.xhtml`.
- Detect common volume marker forms in intro/chapter content and convert them to TOC hierarchy instead of leaving them as prose:
  - `卷一·标题`, `卷一：标题`, `卷一 标题`, `卷一标题`
  - `第一卷·标题`, `第一卷：标题`, `第一卷 标题`
  - `第零卷`, `序卷`, `终卷`
  - title-prefixed forms such as `银河咏叹曲卷四 波拉利斯`
  - Markdown-like forms such as `# 卷二·魔王`
- Volume groups should be top-level and use Chinese numeral labels, for example `卷一·...`, `第一卷·...`, `第零卷·...`, or `终卷·...`.
- Volume numbers must use Chinese numerals, for example `卷一`, `第一卷`, `第零卷`, `终卷`; never `卷1` or `第0卷`.
- If a final volume has a title, prefer `终卷·标题` over a bare title.
- For special parent sections like `尾声·玉满堂`, make them top-level like a volume when the book already has volume groups and the content below that marker is a numbered chapter. Example: in `我和妲己抢男人`, `尾声·玉满堂` is treated as a volume-like parent because prior content is organized into volumes and the content below it is `第56章 紫霄听道`.
- Some readers hide non-linked parent headers. For important parent groups such as `番外`, use a linked parent pointing at the first child if needed.

## Cleanup Rules

- Remove duplicate book title/author boilerplate from intro chapters, such as `《书名》作者：非天夜翔` or standalone `书名 非天夜翔`.
- Remove repeated book title lines from intros when they duplicate EPUB metadata or reader excerpt headers.
- Remove standalone volume-start markers from chapter bodies when the volume is represented in TOC, for example `卷一：鸿渐于陆`, `# 卷二·魔王`, `银河咏叹曲卷四 波拉利斯`.
- After parsing a volume marker, do not leave it in the previous chapter's main body. It may exist as a separate generated volume heading in `nav.xhtml`/`toc.ncx`, but it should not appear as reader body text inside the previous chapter or intro unless the user explicitly wants body volume pages.
- Do not remove prose that merely mentions a volume, such as `第四卷里...`.
- Remove `其他番外` or similar navigation-only marker text from chapter bodies when it is not real content.
- Normalize chapter-title punctuation:
  - Replace bottom dots `.` with middle dots `·` in visible chapter titles and headings.
  - Replace English commas `,` with Chinese commas `，` in Chinese visible chapter titles.
  - Collapse full-width or repeated spaces between `第N章` and title to one normal space.
  - Ensure there is one and only one normal space after the chapter number marker, for example `第12章 标题`.
- When normalizing any visible chapter title, update all three places together: chapter XHTML `<title>`/heading, `EPUB/nav.xhtml`, and `EPUB/toc.ncx`. Do not fix only nav or only toc.
- When the user asks for comma cleanup in the book content, normalize prose too: replace ASCII commas with Chinese commas when the comma is adjacent to Chinese characters or Chinese quotation/bracket punctuation, for example `说道,“` -> `说道，“` and `躺,迟小多` -> `躺，迟小多`.
- Collapse repeated Chinese commas such as `，，` to a single `，`.
- In fanwai chapter titles, remove the current book title if it repeats, for example `相见欢番外...` -> `番外...`. Preserve other referenced book names in crossover titles.

## Numbering Repairs

- If the validator reports duplicate numbered chapters and the first duplicate fills a gap, rename the first duplicate to the missing number across chapter XHTML, nav, and NCX.
- Chapter numbers should stay Arabic in generated titles, for example `第131章 标题`, not `第一百三十一章 标题`, unless the source intentionally uses Chinese numerals and the user asks to preserve it.
- Do not renumber `尾声`, `后记`, or parent group headers unless the user explicitly asks.
- Treat validator warnings from embedded problem headings or source quirks as separate from TOC structure; inspect the chapter text before changing.

## Known Patterns From This Project

- `国家一级注册驱魔师上岗培训通知`: `尾声` and `后记` should come before `番外`; `尾声` before `后记`.
- `我和妲己抢男人`: `尾声·玉满堂` can be a top-level parent section with `第56章 紫霄听道` underneath.
- `破罐子破摔`: if `番外卷元亨利贞` appears in chapter 72, delete it and start `番外` at chapter 73.
- `鹰奴`: final volume title should be `终卷·碰碑`; trailing extras belong under top-level `番外`.
- `清平梦华录`: final volume title is `卷五·吉庆有余`.

## Implementation Notes

Prefer a short Python `zipfile` plus `xml.etree.ElementTree` patch script. Avoid broad regex rewrites of NCX; parse XML and rebuild nav points where possible. Keep namespace registration stable so EPUB files remain parseable.

When reporting completion, list changed books, hierarchy changes, and validation results. Mention any remaining validator warnings as source-numbering quirks only after inspecting them.
