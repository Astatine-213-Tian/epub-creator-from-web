---
name: epub-reformatter
description: >
  Repair and normalize generated EPUB files in this project. Use when Codex needs
  to patch EPUB archives for Chinese novel formatting issues: volume/fanwai/houji
  TOC hierarchy, nav.xhtml and toc.ncx sync, spine ordering, duplicate intro or
  volume marker cleanup, decorative ending marker cleanup and centering, chapter
  title normalization, author/book-name title cleanup, mixed-width Chinese/Latin
  spacing, English-comma spacing, Jinjiang-first package metadata enrichment,
  and EPUB validation.
---

# EPUB Reformatter

## Core Rule

Patch EPUB archives directly and keep every reader-visible surface in sync:

- `EPUB/nav.xhtml`
- `EPUB/toc.ncx`
- affected chapter XHTML files
- `EPUB/content.opf` manifest/spine when adding, deleting, splitting, merging, or moving reading-order files

Before archive surgery, make one overwriteable temp backup per target EPUB under `/private/tmp/epub-creator-from-web-codex-backups/<relative path under epub/>` (for example `/private/tmp/epub-creator-from-web-codex-backups/非天夜翔/锦衣卫.epub`), never beside the EPUB; rewrite to a temporary archive, replace atomically, then validate.

## Workflow

1. Inspect `nav.xhtml`, `toc.ncx`, `content.opf`, and the affected chapter XHTML.
2. Enrich package metadata with `uv run book-enrich-metadata <epub>` or let
   `uv run book-normalize <epub>` run it automatically. Use Jinjiang as the
   authority and KadoKado only after a confirmed Jinjiang miss.
3. Decide hierarchy from actual chapter titles and source markers, not only parser output.
4. Patch chapter files first, then rebuild `nav.xhtml` and `toc.ncx` from the intended hierarchy.
5. If reading order changes, reorder `content.opf` spine to match.
6. Validate:
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
- Merge consecutive multipart fanwai chapters when they share the same base title and differ only by part suffix, for example `第114章 沧浪之龙（一）`, `第115章 沧浪之龙（二）`, etc. The merged chapter title should drop the part suffix, each original part should be separated by a centered and bold divider such as **`（一）`**, **`（二）`**, and all following numbered chapter titles must be renumbered incrementally across chapter XHTML, nav, and NCX. In XHTML, use semantic bold markup such as `<p style="text-align: center; text-indent: 0;"><strong>（一）</strong></p>` while preserving the book's equivalent existing centered style when one is available.
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

- Every finished EPUB needs a reader-visible contents page in the spine, not
  only device navigation metadata. For an EPUB 2 package that has `toc.ncx` but
  no `nav.xhtml`, create `nav.xhtml` as a real XHTML contents page, register it
  in the OPF manifest, insert it in the spine before the prologue or first main
  chapter, and add an OPF guide reference with `type="toc"`. Keep the existing
  NCX as the EPUB 2 device-navigation source.
- For EPUB 3 Apple Books compatibility, keep the navigation vocabulary on the
  canonical literal `epub` prefix: declare
  `xmlns:epub="http://www.idpf.org/2007/ops"` and use
  `<nav epub:type="toc">`. Rewrite serializer-generated aliases such as
  `xmlns:ns1` / `ns1:type`; even though the namespace URI is equivalent XML,
  Apple Books may render an empty Contents panel for that form.
- When a flat EPUB 2 NCX has `序章` or `序言` followed by bare main-chapter
  titles, and every NCX label exactly matches both the linked chapter
  `<title>` and its first heading, add sequential Arabic prefixes beginning
  with `第1章 ` after the prologue. Keep the prologue unnumbered and do not
  number `番外`, `后记`, `尾声`, `终章`, `附录`, or other special sections.
  If the NCX is nested, the labels are mixed numbered/unnumbered, or the
  chapter surfaces disagree, report the ambiguity instead of guessing.
- Center every visible main-chapter and prologue heading. Prefer the book's
  existing centered heading class; otherwise add a minimal inline
  `text-align: center; text-indent: 0` style. This affects the visible heading,
  not ordinary body paragraphs.
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
- In every `番外` chapter title, remove the leading `第N章` marker instead of
  displaying main-text numbering. Apply this to chapter XHTML `<title>` and
  heading text, `nav.xhtml`, and `toc.ncx` together. For example,
  `第152章 番外一承前启后` becomes `番外一·承前启后`.
- Separate a numbered `番外` marker from its following title with one middle
  dot: `番外十一扬帆`, `番外十一 扬帆`, and `番外十一：扬帆` all become
  `番外十一·扬帆`. Keep a bare numbered label such as `番外四` unchanged,
  and preserve an existing correct middle dot.
- Write numbered `番外` markers with Chinese numerals and no intervening
  space: `番外 1 飞天猫`, `番外1·飞天猫`, and `番外 1·飞天猫` all become
  `番外一·飞天猫`; a bare `番外 10` becomes `番外十`.
- In a visible `番外` title, replace a colon immediately after `番外` with a
  middle dot: `番外：蜜月流水账` becomes `番外·蜜月流水账`, and
  `2021 年中秋节番外：前年风月满江湖` becomes
  `2021 年中秋节番外·前年风月满江湖`.
- In a visible Mid-Autumn `番外` title, write `年` after a four-digit year:
  `2022 中秋番外·游园` becomes `2022 年中秋番外·游园`. Apply this only
  when the year is immediately followed by `中秋番外` or `中秋节番外`.
- Remove a single stray terminal separator from a `番外` title instead of
  converting or preserving it, for example
  `2018 年戊戌年中秋番外·啷里个啷.` and
  `2018 年戊戌年中秋番外·啷里个啷·` both become
  `2018 年戊戌年中秋番外·啷里个啷`. Preserve a real ellipsis.
- When normalizing any visible chapter title, update all three places together: chapter XHTML `<title>`/heading, `EPUB/nav.xhtml`, and `EPUB/toc.ncx`. Do not fix only nav or only toc.
- When the user asks for comma cleanup in the book content, normalize prose too: replace ASCII commas with Chinese commas when the comma is adjacent to Chinese characters or Chinese quotation/bracket punctuation, for example `说道,“` -> `说道，“` and `躺,迟小多` -> `躺，迟小多`.
- Collapse repeated Chinese commas such as `，，` to a single `，`.
- In fanwai chapter titles, remove the current book title if it repeats, for example `相见欢番外...` -> `番外...`. Preserve other referenced book names in crossover titles.
- Prettify standalone decorative ending markers when the boundary is obvious: insert middle dots between the book, volume, or section title and terminal words such as `终`, `完`, `正文完`, or `全文完`. Restore missing volume-title separators when the source collapsed them, for example `卷四羽觞醉月终` -> `卷四·羽觞醉月·终`. Normalize marker wrappers made from any number of ASCII hyphens to exactly `——` on each side, for example `--相见欢终--` -> `——相见欢·终——` and `-----卷四羽觞醉月终-----` -> `——卷四·羽觞醉月·终——`. Preserve existing correct separators, for example `——卷四·羽觞醉月·终——`. Center the marker paragraph in the chapter XHTML, using the book's existing centered paragraph style if available or a minimal `text-align: center` style if not. Patch body text only unless the marker also appears in `nav.xhtml` or `toc.ncx`. If the intended boundary is ambiguous, confirm before changing it.

## Metadata Enrichment Rules

- Inspect the live `EPUB/content.opf` first. Treat `<dc:date>` as the
  publication date and `dcterms:modified` as the archive edit timestamp.
- Search Jinjiang by exact `dc:creator`, match the exact `dc:title` on that
  author catalog, and use the established Jinjiang publication date by default.
  A verified Jinjiang date replaces a conflicting local or KadoKado date.
- Use KadoKado only when the Jinjiang author/title lookup completed without a
  defensible match. Do not use KadoKado merely because Jinjiang was temporarily
  unreachable.
- Write official title, creator, `zh-CN`, publication date, source URL,
  description, and ordered subjects. Keep subjects in the order `耽美`, the
  Jinjiang theme/genre, then the Jinjiang time-area category.
- Use the `〖...〗` label attached to a Jinjiang work as its EPUB 3 series name,
  with its position among works bearing that same label. Write
  `belongs-to-collection`, `collection-type=series`, and `group-position`.
- For a locked Jinjiang row, accept a unique masked-title match under the
  verified author, use its published date/source, preserve local description
  and subjects when the source hides them, and report ambiguity instead of
  guessing.
- Patch only `EPUB/content.opf`, refresh `dcterms:modified`, preserve `mimetype`
  as the first uncompressed ZIP member, and make the rewrite idempotent.

## Mixed-Width Spacing Rules

- When the user asks for proper spacing between full-width Chinese and half-width characters, scan parsed visible text across the whole EPUB for the complete defect family, not only the cited example.
- Insert exactly one ASCII space at a direct boundary between a Han character and a half-width Latin letter, Greek letter, or Arabic digit in either direction, for example:
  - `主星VCU07` -> `主星 VCU07`
  - `E7头顶` -> `E7 头顶`
  - `仙女座β星系` -> `仙女座 β 星系`
  - `光纪元20103年` -> `光纪元 20103 年`
- Keep full-width Chinese punctuation flush with the adjacent token, for example `主星 VCU07。”`; do not insert a space before `。`, `，`, `”`, or similar punctuation.
- Preserve punctuation inside half-width tokens such as `B-11`, `F+`, and `γ-B11`. Do not force spaces around hyphens, operators, percent signs, tildes, or other ASCII punctuation merely because they touch Chinese text.
- Preserve ASCII commas in coordinate, tuple, and code-like runs, but ensure exactly one space after each comma: `X337,Y160,Z19 γ-B11` -> `X337, Y160, Z19 γ-B11` and `(0,0,0)` -> `(0, 0, 0)`. Continue to convert an ASCII comma to `，` when it is instead acting as punctuation in Chinese prose.
- For a body-text request, default to chapter paragraph text. If a mixed-width defect occurs in a visible chapter title, update the chapter XHTML title/heading, `nav.xhtml`, and `toc.ncx` together.
- Parse XHTML and inspect text nodes or element text; do not scan raw archive bytes or markup. Preserve inline elements and edit their text/tail nodes without flattening the structure.
- Before rewriting, record match counts and examples by pattern. After rewriting, rerun the same parsed-text audit and require zero direct Han/half-width alphanumeric boundaries and zero ASCII commas without a following space in coordinate/code-like runs.

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

The shared EPUB writers call the single reusable normalizer automatically after
creating a book. Its required order is:

1. Run Jinjiang-first metadata enrichment and record the source, publication
   date, and changed OPF fields in the normalization report.
2. Apply deterministic punctuation, spacing, quote-direction, ad-removal, and
   other context-safe normalization rules.
3. Rescan the normalized XHTML rather than the raw source. Quote findings must
   identify the exact paragraph plus neighboring context; do not use only
   chapter-wide quote totals. Ignore Latin apostrophes such as `I’ll`, and
   accept structurally valid multi-paragraph quotations.
   - Keep trailing `作者有话说` / `作者有话要说` content, but exclude that
     free-form tail from prose anomaly, quote, spacing, ad, and suspicious-
     character review. Continue scanning it for standalone structural markers
     such as `番外卷`.
   - Do not report Unicode emoji or recognized 颜文字 components as bad
     characters. In particular, preserve emoji joiners and Bopomofo-shaped
     glyphs inside forms such as `/(ㄒoㄒ)/~~` or `ㄒ_____ㄒ`; still report the
     same unusual characters when they occur in ordinary prose.
4. Automatically invoke a read-only Codex review for every remaining
   `requires_codex_review` finding. Ask Codex to use Browser Act for suspicious
   characters or corruption when source lookup is needed.
5. Apply only exact, unique, high-confidence text replacements. Do not let the
   review stage perform broad prose rewrites or direct structural archive edits.
6. Normalize and validate again, and write the complete metadata changes,
   automatic fixes,
   review decisions, kept findings, and genuinely unresolved cases to the
   visible report under `reports/normalization/` for books under `books/`.

When reporting completion, list changed books, hierarchy changes, and validation results. Mention any remaining validator warnings as source-numbering quirks only after inspecting them.
