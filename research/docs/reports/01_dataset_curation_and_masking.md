# Dataset Audit: Keep/Drop/Split Decisions

Generated audit artifacts:

```text
generated/style_research/corpus/
  cleaned_manifest.json
  cleaning_report.json
  cleaning_report.md
  duplicate_report.json
  splits.json
  texts/<author>/<title>.clean.txt
```

Reproduce with:

```bash
uv run author-style-research corpus-build --stage clean
```

The script does not modify raw source TXT files under `datasets/raw/`. It writes
generated cleaned copies for style-research use. Run this cleanup stage and verify
residue scans before generating any masked or unmasked chunks.

## Current Result

- Manifest books: 200
- TXT files present: 200
- Missing TXT files: 0
- Authors: 50
- Usable books at `>=50k` cleaned CJK characters: 199
- Primary books at `>=120k` cleaned CJK characters: 196
- Authors with at least 3 books: 50
- Authors with at least 5 usable books: 4
- Exact duplicate cleaned texts: 0

The source TXT files contain many author notes and some scrape/source boilerplate,
but the generated cleaned copies remove detected chapter headings, author-note
blocks, URL lines, and obvious boilerplate lines. A residue scan over the generated
cleaned texts found no hits for the main checked patterns:

```text
作者有话要说
晋江文学城
jjwxc
请收藏
霸王票
营养液加更
最新网址
返回目录
手机阅读
http(s)://
www.
```

## Keep

Keep in the first author-identification benchmark:

- authors with `>=3` usable books;
- books with `>=50k` cleaned CJK characters;
- generated cleaned TXT copies, not raw `datasets/raw/*.txt`;
- raw author/source metadata in `cleaned_manifest.json` for audit and stratified
  analysis.

Use `>=120k` cleaned CJK as the stronger "primary full-length" marker when a method
needs longer stable samples, but do not discard `50k-120k` books from the first
exploratory benchmark.

## Drop From Primary Benchmark

Exclude `非天夜翔/西楚霸王` from the primary chunk benchmark because it has only
38,604 cleaned CJK characters. Keep the three books between 50k and 120k
(`black_di/白日梦`, `black_di/陌上花`, and `酱子贝/我喜欢你男朋友很久了`) in the
usable set but outside analyses that explicitly require the stronger 120k marker.

Authors with fewer than three books were removed from `datasets/`; no one-book or
two-book author remains in the current manifest.

## Split Policy

Use book-level splits only:

```text
train: model fitting
dev: model selection, masking threshold tuning, prompt-method tuning
test: final author-identification report
proxy_transfer: held-out target-author books for style-transfer proxy tasks
```

Never split chunks from the same book across train/dev/test.

Current generated split counts:

- train: 89 books
- dev: 53 books
- test: 53 books
- proxy_transfer: 4 books
- excluded: 1 book

The proxy-transfer holdout currently uses four large target-author books:

- `骑士之歌`
- `天宝伏妖录`
- `万物风华录`
- `清平梦华录`

These should not be used to train the author-style identifier if we use them for
style-transfer method selection.

## Do We Need More Data?

For the first exploratory author-identification benchmark: no, the dataset is enough.

For a robust publishable or production-quality method comparison: yes, more books
would help. The bottleneck is not number of author labels; it is books per comparison
author.

Priority:

1. Add more books for existing comparison authors with exactly 3 usable books.
2. Aim for at least 5 usable books per core comparison author, so each author can
   have more than one training book after dev/test split.
3. Prefer same-domain hard negatives: fantasy, historical, xuanhuan, unlimited-flow,
   modern supernatural, gaming/e-sports, and dialogue-heavy danmei.
4. Add new authors only if each new author can contribute at least 3 usable books.

Current comparison depth is:

- `>=3` books: 50 authors
- `>=5` usable books: 4 authors

Most comparison authors still contribute only three books, so their split remains
one train book, one dev book, and one test book. The 50-author classifier therefore
uses strict book-level generalization; transfer-method selection is restricted to
the much deeper target-author corpus.

## Next Step

After the cleaned corpus passes residue checks, build chunk and masking artifacts
from `generated/style_research/corpus/texts/`:

```bash
uv run author-style-research corpus-build --stage all
```

Current generated chunk artifacts:

```text
datasets/unmasked/chunks.clean.jsonl
datasets/masked/chunks.entity_masked.jsonl
datasets/masked/chunks.entity_masked_v2.jsonl
datasets/masked/chunks.entity_masked_v3.jsonl
datasets/masked/chunks.topic_distorted.jsonl
datasets/masked/chunks.structure_only.jsonl
datasets/masked/mask_terms.json
datasets/masked/masking_report.md
```

Current chunk counts:

- books chunked: 199
- clean chunks: 87,174
- entity-masked chunks: 87,174
- entity-masked v2 chunks: 87,174
- entity-masked v3 chunks: 87,174
- topic-distorted chunks: 87,174
- structure-only chunks: 87,174
- train chunks: 24,957
- dev chunks: 23,683
- test chunks: 35,812
- proxy-transfer chunks: 2,722

Run the mask-quality QA report:

```bash
uv run author-style-research mask-audit
```

Current QA outputs:

```text
generated/style_research/benchmarks/mask_quality_samples.md
generated/style_research/benchmarks/mask_quality_stats.json
```

Current QA result:

- all five chunk views have matching row counts;
- known scrape and author-note residue hits: 0;
- malformed placeholder chunks: 0;
- `entity_masked` is readable enough for human QA but leaked some
  author/book-concentrated names in sample inspection;
- `entity_masked_v2` replaces author-concentrated content terms with `<TERM>`,
  has 90.4% CJK retention, and is retained as the earlier masked view;
- `entity_masked_v3` replaces the same author-concentrated content terms with
  length-preserving `某`, has 100.0% CJK retention, and is the current masked
  view for supervised author-style baselines;
- `topic_distorted` and `structure_only` are diagnostic views, not readable
  training text.

The current reproducible exact n-gram benchmark command is:

```bash
uv run author-style-research authorship-supervised \
  --views clean,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods char_ngrams \
  --classifiers sgd_hinge,sgd_hinge_unbalanced \
  --char-min-df 20 \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20
```

Current benchmark and transfer-research reports:

```text
docs/reports/02_authorship_style_meter.md
docs/reports/03_transfer_iteration1_prompt_methods.md
generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/supervised_author_baseline_results.md
generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/supervised_author_baseline_results.json
```
