# EPUB Creator From Web

Crawl and prepare web novels, then upload drafts to Notion or write local EPUBs directly.

The repository root is the maintained production project. Dataset-heavy
authorship and style-transfer experiments are isolated in the nested
[`research/`](research/) project and never imported by production code.

## Setup

Install `uv`, `mise`, and the GitHub CLI (`gh`), then follow the
[dependency setup](docs/notion-books.md#安装共享依赖). The required private
`notion-books` package needs GitHub repository access and a Go toolchain at build
time, including for local EPUB-only use. The setup installs the pinned toolchain
and Python dependencies; Go is not needed at runtime.

Browser-backed providers require a Chromium-compatible browser. Install Chromium,
Google Chrome, or Playwright-managed Chromium; set `BOOKLIB_BROWSER_PATH` if the
browser is not discoverable automatically.

## Search And Choose A Version

Use `--search` when you know the title but not the best source URL:

```bash
uv run book-to-epub --search "全球高考" --output-format epub
```

The pipeline:

1. Searches supported providers.
2. Builds lightweight previews without downloading full chapter bodies.
3. Ranks results by query match, chapter count, then provider preference.
4. Shows each candidate with title, author, chapter count, first two chapters, and last two chapters.
5. Prompts you to choose one.
6. Crawls the selected provider, prepares its formatting, and writes a Notion draft or a local EPUB, according to the selected output.

Limit search to one provider:

```bash
uv run book-to-epub --search "斗破苍穹" --parser quanben --output-format epub
uv run book-to-epub --search "魔道祖师" --parser mgsf --limit 3 --output-format epub
uv run book-to-epub --search "锦衣卫" --parser xfxs --author 非天夜翔 --output-format epub
```

Automatically choose the top ranked result:

```bash
uv run book-to-epub --search "全球高考" --first --output-format epub -o books/book.epub
```

## Parse A Known Book URL

Pass a supported book URL directly:

```bash
uv run book-to-epub "https://www.mangguoshufang.com/1/2574/info.html" --output-format epub -o books/book.epub
uv run book-to-epub "http://jrkywsy.blog.fc2.com/blog-entry-938.html" --output-format epub -o books/book.epub
```

Choose the output explicitly. Repeat `--output-format` to combine `epub`,
`notion` and `txt`; no destination is selected by default. `--output-format epub`
generates a local EPUB directly. Its default path is `books/<author>/<title>.epub`;
`-o` overrides that path. Z-Library edition downloads require an explicit EPUB
or TXT output and do not enter the web-novel source workflow.

List supported providers when you need to choose one explicitly:

```bash
uv run book-to-epub --list-parsers
```

For provider-specific IDs, force the provider:

```bash
uv run book-to-epub 2574 --parser mgsf --output-format epub -o books/book.epub
uv run book-to-epub doupocangqiong --parser quanben --output-format epub -o books/book.epub
```

## Ingest For Reading And Training

Use `book-ingest` for the normal end-to-end workflow. It crawls the source once,
writes every selected destination: local EPUB, editable Notion draft and/or
TXT. It validates EPUB output and upserts only the requested TXT entry in the
selected dataset manifest. Repeat `--mode` (alias `--output-format`) to combine
destinations. An explicit `-o` or `--txt-output` can select the corresponding
format when no modes are provided.

```bash
# Notion draft
uv run book-ingest "https://www.mangguoshufang.com/1/2574/info.html" --mode notion

# Local EPUB; no Notion upload
uv run book-ingest "https://www.mangguoshufang.com/1/2574/info.html" --mode epub

# Training only
uv run book-ingest "https://www.mangguoshufang.com/1/2574/info.html" --mode txt
```

For all three destinations from one crawl:

```bash
uv run book-ingest "<url>" --mode epub --mode notion --mode txt
```

`--mode both` remains an alias for EPUB + TXT. Training is a use of TXT output;
it has no separate crawler or acquisition skill.
EPUB output defaults to `books/<author>/<title>.epub`. TXT output defaults to
`research/datasets/raw/<author>/<title>.txt`. Dataset metadata is resolved from
the maintained Jinjiang research crawl when available, then falls back to Codex
classification unless `--no-codex-classify` is passed.

For standalone dataset maintenance, use targeted commands:

```bash
uv run book-dataset upsert --txt research/datasets/raw/black_di/替罪羊.txt --author black_di --title 替罪羊
uv run book-dataset export-txt --epub books/black_di/替罪羊.epub
```

Bulk EPUB-to-TXT export is still available, but the EPUB tree must be explicit:

```bash
uv run book-dataset export-txt --books-root books
```

## Ranking

Search results are ranked simply:

1. Query match level.
2. More chapters.
3. Provider preference: `pili45`, then `towasakata`, then `xfxs`, then the rest.

This helps surface fuller versions when the same book exists on multiple sites.

## Provider Notes

- `pili45` and `xfxs` use browser-backed fetching through a Chromium-compatible browser.
- Browser discovery checks `BOOKLIB_BROWSER_PATH`, Playwright-managed Chromium, common `chromium` / `google-chrome` executables on `PATH`, then common macOS app bundle paths.
- To force a browser path:

```bash
BOOKLIB_BROWSER_PATH="/path/to/chromium" uv run book-to-epub --search "全球高考" --parser pili45 --output-format epub
```

- Browser-backed providers may pause on Cloudflare verification.
- `xfxs` native search currently returns a 404 page. When `--author <name>` is supplied, xfxs searches the fixed author page `/a/<GBK-encoded-author>.html` first and skips external site-search for that provider. Without an author hint, xfxs uses external site-search fallback when available.
`src.crawler.search.engines.site_search()` tries DuckDuckGo and raw Google result-page fallbacks. Browser-backed providers can also use the same third-party engines through Chromium when raw search pages throttle. All results are still filtered back to the provider's canonical URL pattern.
- Generated EPUB files belong in `books/<author>/` by default and should not be treated as source code.

## Notion Drafts

This repository uploads editable drafts into the configured
[Notion library](https://app.notion.com/p/3deca693996b810c8774f3658c89a423),
using each book's chapter database and the shared-extra database. Use
`book-notion login` once for MCP OAuth; interrupted uploads resume with `book-notion resume --state <import.json>`.

Automatic native cover upload additionally needs `NOTION_API_TOKEN` in the
process environment; grant that integration access to the target Notion pages.
Only covers use the public API. The scripts do not read `.env` or depend on a logged-in browser.
Local EPUBs retain the cover thumbnail without a cover reading page.

See [output selection, storage, covers and recovery](docs/notion-books.md) and
[shared extras](docs/fanwai-notion.md). Formatting cleanup happens before either
output; the existing presentation renderer interprets only block structure and style.

## Crawl Then Translate

The production bilingual workflow has three passes:

1. English to neutral Simplified Chinese.
2. `content_plan_combined_full_regeneration` author style transfer.
3. English-grounded semantic QA, followed by EPUB construction.

The second pass builds a paragraph-level content plan from English, uses neutral
Chinese only as a terminology/content anchor, then regenerates every paragraph
with retrieved aligned masked examples and a validated style definition. The old
single-author profile/card/cluster reconstruction and flow-repair pipeline is no
longer exposed by `book-translate`; its historical evidence remains under
`research/docs/` and
`research/generated/style_research/`.

Maintained settings live in `book_specs/eternal_gate/config.json`; glossary data
lives in `book_specs/eternal_gate/glossary.json`. The prompt and output schema are
under `book_specs/author_styles/feitianyexiang/`. The local corpus-derived style
asset is ignored by Git and can be reproduced from the frozen research asset:

```bash
uv run --project research author-style-research export-production --overwrite
```

Create an authenticated Patreon crawl snapshot:

```bash
uv run book-crawl "https://www.patreon.com/collection/2218551?view=condensed" \
  --provider patreon \
  --title "永恒之门" \
  --author "顾雪柔" \
  --output generated/crawls/eternal_gate
```

Extract authoritative publisher/creator reply threads into a review artifact:

```bash
uv run book-translate comment-evidence generated/crawls/eternal_gate \
  --config book_specs/eternal_gate/config.json
```

Review `glossary_comment_evidence.json`, then deliberately record confirmed
terminology or protected quotations in the maintained glossary.
Raw comments never enter translation prompts; the semantic translator consumes
only the reviewed glossary and `sentence_translations` data.

Run the complete production pipeline:

```bash
uv run book-translate all generated/crawls/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  --run-dir generated/translation_runs/eternal_gate \
  --run-codex \
  -o books/顾雪柔/永恒之门.bilingual.epub
```

For inspection or resumption, run each stage separately:

```bash
uv run book-translate prepare generated/crawls/eternal_gate \
  --config book_specs/eternal_gate/config.json
uv run book-translate run generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json
uv run book-translate validate generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json
uv run book-translate transfer-style generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  --run-codex
uv run book-translate validate \
  generated/translation_runs/eternal_gate/author_style_transfer \
  --config book_specs/eternal_gate/config.json
uv run book-translate build-epub generated/crawls/eternal_gate \
  --run-dir generated/translation_runs/eternal_gate/author_style_transfer \
  --config book_specs/eternal_gate/config.json \
  -o books/顾雪柔/永恒之门.bilingual.epub
```

Neutral validation always writes `glossary_candidates.json` as a review queue.
It does not mutate the maintained glossary. A candidate becomes canonical only
after review and an explicit edit to the book spec.

The default model order is `gpt-5.6-sol`, `gpt-5.6-terra`,
`gpt-5.6-luna`, `gpt-5.5`, `gpt-5.3-codex-spark`, then `gpt-5.4`.
The runner advances only after an explicit capacity, quota, unsupported-model,
or availability failure. Contract and fidelity failures retry on the same model
and recursively bisect the block; a single paragraph can fall back to the
validated neutral text only after the style output fails deterministic fidelity
checks and all non-fidelity checks pass.

`validate --config ...` automatically runs semantic QA for author style-transfer
runs. The QA stage reviews the highest-risk deterministic candidates, applies
only low-risk repairs that pass acceptance checks, and validates repaired output
again. EPUB construction requires a current, non-dry-run QA summary bound to the
styled outputs. Any failed style chunk, failed QA batch, or rejected QA candidate
stops the production command. A `true_loss` verdict must be repaired and accepted;
unresolved true-loss findings also block EPUB construction.

The same three-pass pipeline can be used for another book by providing a config,
glossary, and hash-locked author-style asset for that target author.

## Author Style Research

Author identification, corpus masking, style-meter evaluation, and transfer
experiments are isolated in the nested [`research/`](research/) project. Its
copyrighted corpora and generated evidence stay under `research/datasets/` and
`research/generated/`; production code does not import research modules.

```bash
cd research
uv sync
uv run author-style-research verify
```

The research project has its own dependency lock and imports shared crawler and
snapshot primitives through an editable dependency on the production project.
See [`research/README.md`](research/README.md) for the active reports and
reproduction entry points.

## Book Workflow Skill

Use the single [book-management skill](.agents/skills/book-management/SKILL.md)
for adding, updating or repairing books. It asks for missing output choices and
routes to crawler-development or formatting references only when needed.
A training request selects TXT, the requested dataset path and a targeted
manifest update in this same workflow.

## Architecture

The [architecture guide](docs/architecture.md) describes module ownership,
input/output contracts and dependency boundaries. The main modules are:

- `src/crawler/`: providers, search, fetching and reusable snapshots.
- `src/content/`: shared source models, cleanup and formatting preparation.
- `src/epub/`: local EPUB presentation, validation and edition repair.
- `src/notion/`: Notion draft upload, storage adaptation and cover upload.
- `src/translation/`: translation, style transfer and semantic QA.
- `src/workflows/`: compose those modules and select the output destination.

`src/dataset/`, `src/metadata/` and `src/runtime/` own dataset manifests,
metadata lookup and runtime utilities respectively. CLI modules parse arguments
and call these modules; providers never choose output destinations.

## Validation

Useful checks after parser or search changes:

```bash
uv run book-to-epub --list-parsers
uv run book-to-epub --search "known title" --parser provider_name --output-format epub
uv run python -m compileall -q src
uv run python -m unittest discover -s tests -p 'test_*.py'
```
