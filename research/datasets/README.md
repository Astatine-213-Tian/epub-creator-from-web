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
external_*/             held-out construct and author challenges
```

`raw/` is the recoverable source of truth. `unmasked/` and `masked/` are derived
by `workflows.audit_style_dataset`. The canonical classifier uses strict book-level
splits from `generated/style_research/corpus/splits.json`; chapters from one book
must not cross train, development, and test roles.

Rebuild derived chunks from `research/`:

```bash
uv run author-style-research corpus-build --stage all
uv run author-style-research mask-audit
```

Do not normalize or rewrite the frozen manifest merely to change path spelling:
several later preregistrations bind its byte hash. Dataset readers support the
historical path forms after the repository move. New ingest entries use paths
relative to this dataset root.

## Dataset Registry

| Dataset | Contents | Experimental role | Retention |
| --- | --- | --- | --- |
| `raw/`, `unmasked/`, `masked/` | Canonical 50-author corpus and derived views | Training, development, and book-held-out authorship evaluation | Keep; canonical research data |
| `external_author_challenge_v2/` | 10 books from 5 authors absent from the 50-author corpus; every row is marked `excluded` | External false-positive/specificity challenge for CR-FYSM-v4; never training data | Keep; its failed confirmation result is retained evidence |
| `external_target_construct_v1/` | 26 chunks from the short target-author book `西楚霸王`; every row is marked `excluded` | Abandoned target-construct pilot; it was never used by a retained benchmark or report | Removal candidate; its cleaned text already exists byte-for-byte in the canonical corpus |

The external-author challenge produced 95.4% overall specificity but failed its
registered per-author, per-book, and bootstrap gates. That failure is part of
the reason CR-FYSM-v4 was rejected as a transfer-selection endpoint, so the
challenge corpus remains necessary for a full audit even though it is not used
by the production method.
