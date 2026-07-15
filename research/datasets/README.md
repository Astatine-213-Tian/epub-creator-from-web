# Research Datasets

This directory is local and ignored by Git because it contains copyrighted book
text. It is owned by the research project even though the parent production
`book-ingest` and `book-dataset` commands write new entries here.

```text
dataset_manifest.json   book metadata and corpus membership
raw/                    source TXT exports; never modified by cleaning
unmasked/               cleaned chunk views
masked/                 entity-masked and diagnostic chunk views
provenance/             source-repair and replacement records
external_author_challenge_v2/
                        held-out authors used only for specificity testing
```

`raw/` is the recoverable source of truth. Cleaning never edits it. Full cleaned
book copies live under `generated/style_research/corpus/texts/` because they are
rebuildable artifacts, while `unmasked/` and `masked/` contain the aligned chunk
views consumed by experiments. All are derived by
`workflows.audit_style_dataset`. The canonical classifier uses strict book-level
splits from `generated/style_research/corpus/splits.json`; chapters from one book
must not cross train, development, and test roles.

The current masking contract fits its global 12,000-term content vocabulary on
training books only, applies that fixed vocabulary to every split, and optionally
adds per-book terms selected without author labels. `train_global_masked` is the
global-only ablation; `entity_masked_v3` is the global-plus-local research view.

Rebuild derived chunks from `research/`:

```bash
uv run author-style-research corpus-build --stage all
uv run author-style-research mask-audit
```

The active manifest is the current corpus contract. Historical preregistrations
whose manifest hash no longer matches remain snapshots and must not be used as
current results. New ingest entries use paths relative to this dataset root.

## Dataset Registry

| Dataset | Contents | Experimental role | Retention |
| --- | --- | --- | --- |
| `raw/`, `unmasked/`, `masked/` | Canonical 50-author corpus and derived views | Training, development, and book-held-out authorship evaluation | Keep; canonical research data |
| `external_author_challenge_v2/` | 10 books from 5 authors absent from the 50-author corpus; every row is marked `excluded` | External false-positive/specificity challenge for CR-FYSM-v4; never training data | Keep; its failed confirmation result is retained evidence |

The external-author challenge produced 95.4% overall specificity but failed its
registered per-author, per-book, and bootstrap gates. That failure is part of
the reason CR-FYSM-v4 was rejected as a transfer-selection endpoint, so the
challenge corpus remains necessary for a full audit even though it is not used
by the production method.
