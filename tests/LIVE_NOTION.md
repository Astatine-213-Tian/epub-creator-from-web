# Live Notion upload test

Run `tests/live_notion_upload.py` with the crawler environment after its ordinary
unit tests pass. It injects a synthetic `CrawledBook` at the collection boundary,
then executes the real source preparation, metadata mapping, duplicate scan,
template application, chapter/extra upload, checkpoint persistence and resume.
Live readback checks names and complete formatting/content. It uses the crawler's
own OAuth login and public `book_specs/notion/config.json` database settings.

This tests the result-to-Notion path. Provider crawling/parsing remains covered
by provider fixtures and the regular crawler suite. It does not run CMS publishing
or require the NAS checkout. The CMS owns its independent Notion-to-Bookshelf test.

With authorization to create and clean up live test pages, run from this repo:

```bash
live_state=$(mktemp -d /private/tmp/crawler-notion-live.XXXXXX)
uv run python tests/live_notion_upload.py import --state "$live_state"
```

The configured database/template must already exist. The test creates uniquely
marked authors/books/extras; it never publishes. The synthetic book has no cover,
so native cover upload requires its separate test. State/checkpoints remain in
`live_state` and can diagnose a failure; never retry an ambiguous creation by
throwing away its checkpoint. Machine-wide unsupported SOCKS proxy settings may
need to be unset for the command.

After success or failure, move only the test work (including its chapter database),
author and created extra from `run.json` and its `import_state` (when present) to
Notion Trash. If a create response was lost, locate the pages by the unique
`marker` in their names; cleanup verification also checks those names. The MCP transport has no page-trash operation, so use the Notion UI.
Then check the active inventories:

```bash
uv run python tests/live_notion_upload.py verify-cleanup --state "$live_state"
```

Keep any useful report, then delete the isolated test state. Credentials remain
in the crawler's own local credential store.
