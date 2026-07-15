# Eternal Gate Author-Style Transfer Research Report

Generated from frozen artifacts at `2026-07-13T00:19:02+00:00`.

## Current Status: Iteration 2

Iteration 2 has now tested three aligned pseudo-parallel transfer methods with `gpt-5.5` on a fresh 36-row screen. All three methods increased mean target-author margin in both benchmark arms, but none passed the preregistered joint gate because own-author hard-fidelity failures remained above 10%. The reused binary threshold also has stale iteration-1 allocation/protocol bindings, so no threshold-based accuracy or 80% result can be claimed. The 116-row confirmation set remains unopened.

Detailed procedures, method definitions, per-arm results, graphs, retry statistics, and the independent evaluator's findings are in [the iteration-2 aligned-pairs report](04_transfer_iteration2_aligned_pairs.md).

## Iteration 1 Executive Result

The author-identification prerequisite passed, but the first registered style-transfer screen did not. The exact masked character n-gram meter reaches **87.8%** 50-author test accuracy and **83.6%** balanced accuracy, yet none of the 11 non-control transfer methods met the joint screening gate. The frozen promotion decision is **`no_method_qualified`**, so confirmation and final validation have not started.

| Stage | Data | Procedure | Result | Status |
| --- | --- | --- | --- | --- |
| 0. Corpus cleanup | 200 books, 50 authors | Clean, deduplicate, book-level split, six chunk views | 199 usable books chunked; 87,174 chunks; zero checked residue hits | complete |
| 1. Author-style meter | 24,957 train / 23,683 dev / 35,812 test chunks | Compare n-grams and interpretable classifiers on clean and entity-masked text | Selected exact 2-4 character n-gram class-balanced hinge model: 87.8% test accuracy | complete |
| 2. Generated-domain calibration | 32 held-out target passages | Original target Chinese versus production-compatible neutral Chinese | Threshold `0.218175`; sensitivity 100%; neutral FPR 0% | complete |
| 3. Proxy screening | 24 own-author + 12 cross-author chunks | Neutral control plus 11 transfer interventions, one frozen output per method/sample | 396/396 non-control outputs valid; all methods 0% deterministic style success | complete, failed gate |
| 4. Independent outcome audit | Frozen evaluation and promotion artifacts | Separate evaluator checks rule compliance and interpretation | NO-GO; empty shortlist is correct | complete |
| 5. Refinement / repair | Would use shortlisted methods only | Lighter intensity and one critique repair | Not run because shortlist is empty | stopped by protocol |
| 6. Confirmation | 104 own-author + 48 cross-author rows | Exact promoted methods only | Not opened | locked |
| 7. Final validation | Four transfer-held-out target books; 80 rows planned but not yet sampled | One-time transfer replication after confirmation passes | Row IDs do not yet exist | locked |

## Research Question

Can a controlled second-pass rewrite move production-compatible neutral Chinese toward the prose style of 非天夜翔 while preserving the English source meaning, natural Chinese readability, paragraph structure, and no-copy constraints?

The intended production pipeline remains:

```text
Eternal Gate English
  -> pass 1: neutral Simplified Chinese (English is semantic authority)
  -> pass 2: tested author-style transfer
  -> pass 3: English-grounded semantic/readability validation and repair
```

No pass-2 method is approved for production yet.

## Experimental Design and Data Lineage

The **experimental unit is one contiguous prose chunk**, normally about 1,500 cleaned CJK characters. Book is the primary independence boundary: chunks from the same book never cross corpus train/dev/test splits. Confirmation inference is registered at the book or author cluster level rather than treating thousands of correlated chunks as independent observations.

| Data role | Authors / books | Units | May influence | Access status |
| --- | --- | ---: | --- | --- |
| Classifier fit | 50 authors / 89 train books | 24,957 masked chunks | Vectorizer vocabulary and classifier weights | used |
| Classifier development | 50 authors / 53 dev books | 23,683 masked chunks | Diagnostic model comparison | used |
| Classifier book-disjoint benchmark | 50 authors / 53 test books | 35,812 masked chunks | Reported author-classifier metrics only | used; not available to transfer prompts |
| Transfer evidence pool | 1 target author / 29 target train books, contrasted with 49 authors | 5,650 target + 19,307 comparison train chunks | Cards, retrieval, close reading, and style-meter weights | used |
| Proxy benchmark pool | 13 authors / 20 books | 220 sampled chunks | Calibration, screening, or confirmation according to frozen IDs | allocated |
| Generated-domain calibration | 1 target author / 8 proxy books | 32 chunks, 4 per book | One frozen target-margin threshold | used once |
| Iteration-1 screening | 13 authors / 20 books | 24 own-author + 12 cross-author chunks | Method-family promotion decision | used |
| Iteration-1 confirmation (historical) | same 13 authors / 20 books | 104 own-author + 48 cross-author chunks | Registered iteration-1 endpoint; superseded by the 116-row iteration-2 allocation | unopened in iteration 1 |
| Final validation | 1 target author / 4 books held out from transfer development | Planned 80 chunks, 20 per book; exact row IDs do not yet exist | One-time transfer replication; books contributed to Stage 1 classifier benchmark metrics | book reservation only |
| Eternal Gate production input | English novel; no Chinese target exists | 0 research rows | Nothing in method selection | excluded from all training and benchmark sets |

No style-transfer LLM was fine-tuned in iteration 1. Here, **training evidence** means corpus material used to fit the style meter or construct prompts; **development data** means the frozen proxy books used for calibration/screening/confirmation; and **final test data** means the four target-author books reserved for one-time validation. These roles are kept separate throughout the report.

### Information available at each generation step

| Step | Inputs visible to that step | Inputs deliberately hidden | Output |
| --- | --- | --- | --- |
| Benchmark English construction | Held-out original Chinese, paragraph IDs | Method label and later score | Synthetic plain-English semantic source |
| English QA/repair | Original Chinese plus synthetic English | Style-transfer output | Approved or selectively repaired English |
| Neutral Chinese generation | Approved English plus paragraph IDs | Original Chinese, author/book identity, style evidence | Production-compatible neutral Chinese |
| Style-transfer generation | Approved English, byte-identical neutral Chinese, and registered method evidence | Original Chinese, book title, sampling stratum, evaluation role, scores | One candidate Chinese rewrite |
| Deterministic evaluation | Candidate, neutral control, frozen scorer, hidden allocation, registered references | No adaptive prompt changes | Style scores and hard surface/copy flags |
| Independent semantic/readability judgment | English source and blinded neutral/candidate pair | Method identity and target original | Paired semantic/readability findings; only for deterministic survivors |

## Stage 0: Corpus Construction

### Source corpus

The current corpus contains **200 books by 50 authors**. Every retained author has at least three books. Cleaning operates on copies under `generated/style_research/corpus/`; raw TXT files under `datasets/raw/` are not modified. The cleaner removes chapter headings, author-note blocks, URL lines, and known source boilerplate. Exact cleaned-text duplicates: 0.

| Split | Authors | Books | Chunks | Purpose |
| --- | ---: | ---: | ---: | --- |
| train | 50 | 89 | 24,957 | Fit classifiers and derive transfer evidence; only this split may supply target examples/cards |
| dev | 50 | 53 | 23,683 | Classifier development diagnostics and four target proxy books |
| test | 50 | 53 | 35,812 | Book-disjoint 50-author classifier benchmark; four target books reserved from transfer development |
| proxy_transfer | 1 | 4 | 2,722 | Additional target-author proxy reconstruction books |
| excluded | 1 | 1 | 0 | Too short for primary chunk benchmark |

Chunks target 1,500 cleaned CJK characters with an 800-character minimum. Chunks from one book never cross train/dev/test boundaries. This prevents random chapter mixing from leaking book-specific language into evaluation.

### Target-author book allocation

- **Style/card training, 29 books:** 国师帮帮忙, 天之战记, 二零一三（末日曙光）, 天地白驹, 灵魂深处闹革命, 大设定师, 鹰奴, 飘洋过海中国船, 银河咏叹曲, 王子病的春天, 破罐子破摔, 逆世界之书, 理想之城, 武将观察日记, 北城天街, 金牌助理, 江湾路七号男子宿舍, 幺儿, 锦衣卫, 江东合伙人, 我和妲己抢男人, 已枯之色, 放开那个受, 朝圣, 将军们的情书, 战七国, 江东双璧, 别过来（仙境幻想游记）, 星盘重启.
- **Development proxy, 4 books:** 相见欢, 山有木兮, 乱世为王, 图灵密码.
- **Additional proxy-transfer, 4 books:** 骑士之歌, 天宝伏妖录, 万物风华录, 清平梦华录.
- **Transfer-held-out final-validation book reservation, 4 books:** 星辰骑士, 夺梦, 定海浮生录, 国家一级注册驱魔师上岗培训通知. The exact 80 rows are deliberately created only after a confirmation method passes and is locked.
- **Excluded for length:** 西楚霸王.

### Content masking

`entity_masked_v3` replaces names and author-concentrated content terms with length-preserving `某` spans. It retains 100% CJK length, has row parity with clean chunks, and has no malformed placeholder chunks. The masked view is used for the classifier and retrieval evidence to reduce theme, character, and book-identity learning. Masking is a control, not proof that all content signal has been removed.

## Stage 1: Author-Style Meter

All classifier variants fit feature transforms and model weights on the same 24,957 `train` chunks. The 23,683 `dev` chunks and 35,812 `test` chunks are book-disjoint from training. Because several classifier families were iteratively compared using their reported test results, `test` is best interpreted as a **book-disjoint benchmark**, not a pristine selection-blind estimate. The four final books remain unopened to transfer generation, threshold calibration, and transfer-method selection, but their chunks did contribute to Stage 1 classifier benchmark metrics. Final validation is therefore a transfer-held-out replication, not a fully end-to-end selection-blind test.

### Methods compared

| Feature family | What it measures | Best masked chunk accuracy | Research role |
| --- | --- | ---: | --- |
| Exact character n-grams | Auditable TF-IDF over exact Chinese 2-4 character sequences | **88.2%** unweighted; **87.8%** class-balanced | Primary style meter |
| Hashed character n-grams | Approximate 2-4 character n-grams in 262,144 hash buckets | 85.0% | Faster approximation; rejected as final meter |
| Combined interpretable + richer function words | Punctuation, dialogue, length, function characters, multi-character function words | 69.5% | Diagnostic guardrail |
| Function words + characters | Chinese function phrases plus high-frequency function characters | 60.0% | Grammar/function-word diagnostic |
| Function characters only | Single-character low-semantic inventory | 45.0% | Weak interpretable baseline |
| Chinese function words only | Multi-character connective, aspect, particle, deixis, and frame terms | 39.4% | Weak interpretable baseline |
| Punctuation/dialogue only | Quote, dialogue, punctuation, and related rates | 21.5% | Diagnostic only |
| Sentence/paragraph length only | Sentence and paragraph shape statistics | 16.7% | Diagnostic only |

### Selected meter

The selected proxy is the **class-balanced SGD hinge classifier with exact character 2-4 grams on `entity_masked_v3`**, `min_df=20`, maximum 80,000 features. It was fit on train books and evaluated on book-disjoint benchmark books. The fit contains 5,650 target-author chunks and 19,307 comparison-author chunks; class balancing reduces the effect of that unequal training support.

| Metric | Result |
| --- | ---: |
| 50-author test chunk accuracy | 87.8% |
| Balanced accuracy | 83.6% |
| Macro F1 | 82.3% |
| Book-majority accuracy | 90.6% |
| Target-author F1 | 85.9% |
| Target-author recall | 100.0% |
| Majority-class baseline | 6.0% |

This is a **proxy meter**, not a human style judgment and not a probability model. The raw hinge margin is reported as a decision margin. Interpretable features remain secondary guardrails because they are easier to understand but materially less accurate.

## Stage 2: Iteration-1 Proxy Benchmark Data

### Why the source passes through English

The actual Eternal Gate input is English, so the benchmark reproduces that information bottleneck instead of merely neutralizing Chinese directly:

```text
held-out original Chinese (hidden from generator)
  -> LLM-generated plain-English benchmark source
  -> independent semantic QA and selective repair
  -> production-compatible neutral Chinese
  -> style-transfer candidate
  -> evaluation against English, neutral, and hidden original
```

The benchmark English is **synthetic English reconstructed from held-out Chinese**, not the published Eternal Gate translation. This design gives every proxy row a recoverable English semantic source while matching the production direction. It does not test quirks unique to Eternal Gate's actual English translator, so a later external validation sample must use real Eternal Gate English before production adoption.

The neutral prompt is version `neutral_translation.v1` and is pinned as the core `eternal_gate_pass_1` contract. It preserves paragraph IDs and details, uses natural neutral Simplified Chinese, and forbids author imitation, style cards, examples, and embellishment. Production may add glossary context, but changing these core neutrality rules requires a new benchmark version.

### Frozen 220-row development allocation

| Arm | Books/authors | Rows | Purpose |
| --- | --- | ---: | --- |
| Own-author reconstruction | 8 target books x 20 chunks | 160 | Can the pipeline recover author signal lost through English and neutral Chinese? |
| Cross-author transfer | 12 other authors x 1 book x 5 chunks | 60 | Can target style transfer to different content instead of reconstructing residual target content? |

The 160 own-author rows are divided into 32 calibration rows and 128 method-evaluation rows. The first registered screen uses 24 method rows: three from each target book. The cross-author screen uses one row from each of the 12 authors below.

| Own-author screening books (3 rows each) | Cross-author screening books (1 row each) |
| --- | --- |
| 骑士之歌, 天宝伏妖录, 万物风华录, 清平梦华录, 相见欢, 山有木兮, 乱世为王, 图灵密码 | priest/杀破狼; 木苏里/全球高考; 巫哲/轻狂; 唐酒卿/南禅; 墨香铜臭/魔道祖师; 淮上/破云; 梦溪石/无双; 西子绪/我五行缺你; 酱子贝/放学等我; 稚楚/营业悖论[娱乐圈]; 莫晨欢/第四视角; 漫漫何其多/当年万里觅封侯 |

The frozen sample selector balances book position, six heuristic scene types, length, punctuation, sentence flow, and dialogue density, with a minimum within-book chunk-index distance of three. The generator sees opaque sample IDs, English, and neutral Chinese. It does not see book title, author, sampling stratum, original Chinese, or evaluation role.

### Frozen sample artifact chain

| Artifact | Contents and access role |
| --- | --- |
| `sample_sets/development_proxy_v1.runner_manifest.jsonl` | Opaque sample IDs and generated-artifact paths; this is the only sample manifest available to neutral/style runners |
| `sample_sets/development_proxy_v1.evaluator_allocation.jsonl` | Author, book, arm, research role, and sampling strata; evaluator-only |
| `sample_sets/development_proxy_v1.hidden_targets.jsonl` | Original Chinese target text keyed by opaque ID; evaluator-only |
| `sample_sets/development_proxy_v1.screening_v1_ids.json` | Exact 36-row iteration-1 screening cohort |
| `sample_sets/development_proxy_v1.confirmation_v1_ids.json` | Exact 152-row iteration-1 confirmation cohort; historical and not the current 116-row iteration-2 allocation |
| `sample_sets/development_proxy_v1.summary.json` | Counts, book/author roster, canonical row hashes, and leakage-control declarations |
| `protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json` | Raw file-byte SHA-256 bindings for the active frozen analysis |

The summary's `runner_manifest_sha256`, `evaluator_allocation_sha256`, and `hidden_targets_sha256` are hashes of normalized, key-sorted JSON rows. The active analysis lock stores raw file-byte hashes, so those values are expected to differ. Running `style_transfer_research.py validate` recomputes both the row integrity and leakage checks; the current 220-row sample set passes with zero target-metadata, hidden-target, reserved-book, residue, or spacing leakage.

### English-source QA and repair

English source artifacts exist for all 220 rows, but the calibrated screening cohort uses 68 rows: 32 calibration plus 36 screening. Every selected English source is checked by a separate stateless Chinese-to-English semantic QA request. A repair round regenerates only the still-failing English sources from the prior QA findings, then sends them through a new QA request. It does not change Chinese style and does not repeatedly edit already approved rows.

| QA point | Rows entering repair | Rows still unresolved after QA |
| --- | ---: | ---: |
| Initial source QA | 68 | 41 |
| Repair round 1 | 41 | 19 |
| Repair round 2 | 19 | 13 |
| Repair round 3 | 13 | 8 |
| Repair round 4 | 8 | 4 |
| Repair round 5 | 4 | 3 |
| Repair round 6 | 3 | **0** |

After source approval, neutral generation completed 68/68 rows. Calibration consumes 32; the style-transfer screen consumes the remaining 36. The initial QA and six repair/QA rounds are recorded under `runs/development_proxy_v1/proxy_v1_gpt54_20260711_v2/ledgers/` as `english_source_qa.jsonl` and `english_source_repair_qa.round_01.jsonl` through `round_06.jsonl`.

## Stage 3: Style-Meter Calibration

Calibration compares the hidden original target passage (positive) with its neutral translation (negative) on the same 32 rows. It enumerates observed margins, keeps thresholds with at least 80% original sensitivity and at most 10% neutral false positives, maximizes balanced accuracy, and chooses the higher threshold on ties.

| Calibration field | Frozen value |
| --- | ---: |
| Rows | 32 |
| Margin threshold | `0.218174636` |
| Original sensitivity | 100.0% |
| Neutral false-positive rate | 0.0% |
| Balanced accuracy | 100.0% |

Method-evaluation rows are forbidden from recalibrating this threshold.

## Stage 4: Registered Transfer Methods

All requests used `gpt-5.4` with `high` reasoning, one ephemeral session per sample, a macOS Seatbelt profile denying repository reads/writes, and at most two schema-generation attempts. English and neutral inputs are byte-identical across methods. Every non-control method generated 36/36 valid outputs, for 396 method outputs total.

### Generation provenance

| Component | Frozen version | SHA-256 |
| --- | --- | --- |
| English semantic source | `english_semantic_source.v1` | `03a79e0b0de1df5b9721f81eb967aee49f9f0b13a482647fefe15dae115b21fa` |
| English source QA | `english_source_qa.v1` | `c8243b5e39361e79466484a90c68746011ff6b78a3fa7acf3ef11e59f32648d5` |
| English source repair | `english_source_repair.v1` | `13bd8b500ea1ba9f590dfa81b3acb328304800fc7e4d8ee57900b6639d57d061` |
| Neutral translation | `neutral_translation.v1` / `eternal_gate_pass_1` | `600cc82d5f40ec70b667b918f5c5cfebad1b8378212e010b49c4ebf2847ad43b` |
| Style transfer | `style_transfer_method.v1` | `d3e9d7e98c3adfd8c601768ee1f735f7c5c186d881cb57ec23ca4d6b4c85c058` |

The run ledger records prompt/input/output hashes, response IDs, model name, Codex CLI version, runner hash, and sandbox profile. The provider exposes no temperature and no replayable backend snapshot; therefore artifacts provide provenance and deterministic reevaluation of frozen outputs, but not guaranteed byte-identical future LLM regeneration.

### Tested interventions

| Method / tested intensity | What was tested | Training evidence exposed to the method |
| --- | --- | --- |
| `neutral_only:none` | No style rewrite. The production-compatible neutral Chinese draft is scored unchanged. This is the negative control and the paired baseline for every lift value. | English semantic source plus neutral Chinese only. |
| `generic_author_style_light:light` | Author-name prompt, minimal rewrite. The model sees the target author name but no style cards or examples. It is told to make only a light, source-supported recast and to skip weak opportunities. | Target author name only; no corpus excerpts are supplied. |
| `generic_author_style_strong:strong` | Author-name prompt, stress-test rewrite. The same author-name-only cue is applied assertively as a stress test. This tests whether stronger generic prompting raises style strength at the cost of fidelity. | Target author name only; stronger rewrite-scope instruction. |
| `global_style_cards:medium` | Corpus-derived global style rules. The model receives compact executable rules derived from 29 target-author training books against 49 comparison authors. Rules cover dialogue tags, punctuation, function words, and sentence flow, each with a semantic guardrail. | Aggregate statistics from target train books and the 49-author train corpus. |
| `scene_routed_style_cards:medium` | Scene-specific corpus rules. The neutral draft is routed to scene types such as dialogue, action, reflection, care, travel, or exposition. Only cards registered for the inferred scene are supplied. Routing uses neutral Chinese, never hidden target metadata. | Scene-conditioned cards derived from target and comparison train books. |
| `sentence_flow_cards:medium` | Sentence and paragraph rhythm rules. The intervention isolates flow: short-sentence share, mean sentence length, comma-linked clauses, and paragraph progression. It tests whether cadence alone carries enough author signal beyond vocabulary. | Target-versus-comparison sentence-flow statistics from train books. |
| `function_word_punctuation_dialogue_cards:medium` | Interpretable grammar and dialogue rules. The intervention isolates lower-content cues: Chinese function-word families, punctuation, plain speech tags, laughter tags, and micro-reactions. Each rule prohibits adding unsupported emotion, motive, causality, or action. | Interpretable feature statistics from target and comparison train books. |
| `retrieved_examples_only:medium` | Nearest target-author examples. For each neutral draft, retrieval returns three entity-masked target-author train chunks of at most 450 CJK characters. The model gets examples but no explicit style definition. | Three entity_masked_v3 examples; train split only; near duplicates excluded. |
| `scene_cards_plus_examples:medium` | Scene rules plus target examples. This hybrid combines scene-routed executable cards with three retrieved masked examples. It tests whether explicit constraints and tacit demonstrations are complementary. | Scene cards plus three masked target-author train examples. |
| `contrastive_examples:medium` | Target examples versus hard negatives. The model receives three target-author examples and matched hard-negative examples from other authors. It is told to infer only structural differences shared by target examples and absent from negatives, reducing genre/topic imitation. | Masked target and hard-negative train examples from the 50-author corpus. |
| `llm_close_reading_style_definition:medium` | LLM close-reading definition. A separate close-reading pass synthesizes discourse, grammar, narrator stance, and scene-mechanics observations from masked target and comparison passages. The transfer model receives that qualitative definition without aggregate statistical gates. | Close reading of 78 masked train passages: 29 target and 49 comparison. |
| `llm_close_reading_cards_statistical_gates:medium` | Close reading constrained by statistics. This hybrid combines the LLM close-reading definition with corpus-derived cards and requires qualitative claims to agree with measured target-versus-comparison tendencies. | Masked close reading plus statistically supported train-corpus cards. |

### Exact method and payload bindings

The family descriptions above are explanatory. The table below identifies the exact machine-readable config and per-intensity payload hash used to build each request.

| Method / intensity | Family | Config path | Config SHA-256 | Payload SHA-256 |
| --- | --- | --- | --- | --- |
| `neutral_only:none` | `control` | `generated/style_research/style_transfer_experiments/method_registry/methods/neutral_only.v1.json` | `aa836e8507a2f65b0fa61a029ec1a532c902170f520a0a1ff3a9671d7a4aed42` | `f5be834a2b4e060a97ea97aa829921dd44c0749a4c4bc3eacea8aacf4603274c` |
| `generic_author_style_light:light` | `generic_prompt` | `generated/style_research/style_transfer_experiments/method_registry/methods/generic_author_style_light.v1.json` | `ffc1abed0e45000c4db5a5bcf60b80d32c011cdfb9a2e9861060ae9d8540fa69` | `c656f398f2dccc8309bd1b179235125b6c7ab213e931ef017d36d08ddb273825` |
| `generic_author_style_strong:strong` | `generic_prompt` | `generated/style_research/style_transfer_experiments/method_registry/methods/generic_author_style_strong.v1.json` | `c7bb0e4f21fecf29228ffd7ea7b9731e90384c11a02c93a5a56e5ebc51732d68` | `53fc4ef95ab45f70d77094d2952b6999ea43a3277af3dfd3b069ddf8b89728a6` |
| `global_style_cards:medium` | `structured_definition` | `generated/style_research/style_transfer_experiments/method_registry/methods/global_style_cards.v1.json` | `3ddbc0f12b091ec7f9ace87b393d535888dc2833e0aafdc2dd0c4561d505be10` | `279a1e8d1f9ce66bdd0006479bd7d6a57cbba71fd401054d4420994e2cca22ed` |
| `scene_routed_style_cards:medium` | `structured_definition` | `generated/style_research/style_transfer_experiments/method_registry/methods/scene_routed_style_cards.v1.json` | `7bb05a5e07a1b92c146b471e8c9b77ae8a3a92bd700f6e1770d3b111afd00b6e` | `77ae66ed2dc55a9c1111fd82f44e6e94a29cfb66dbbe3a1d01475bcd7879b57e` |
| `sentence_flow_cards:medium` | `structured_definition` | `generated/style_research/style_transfer_experiments/method_registry/methods/sentence_flow_cards.v1.json` | `c2bc1f81362c6aaa5364c5252f57f2afdd76d76de86b635803405887c833ef09` | `3e11adf64bf481f43c0edb9de31090f52cb3a5c13d7332d692fb8dfd37cdb69e` |
| `function_word_punctuation_dialogue_cards:medium` | `structured_definition` | `generated/style_research/style_transfer_experiments/method_registry/methods/function_word_punctuation_dialogue_cards.v1.json` | `2b6018047afafbd85399cb6d5d995561aa7f78518a764500572da0aafbe61810` | `9bfa355da0563db60da92b89d79298a99bd71b3cb39cd74858803dc6fd637652` |
| `retrieved_examples_only:medium` | `retrieval` | `generated/style_research/style_transfer_experiments/method_registry/methods/retrieved_examples_only.v1.json` | `6eb5387569e5a41d270bf4802deb59cd793490f701bd5b2f3212ad607200089d` | `497dd43c700fab82f49f4c78eee659aaba4fa7443f39aa4591326930eb4c6d41` |
| `scene_cards_plus_examples:medium` | `hybrid` | `generated/style_research/style_transfer_experiments/method_registry/methods/scene_cards_plus_examples.v1.json` | `bed918734fee72a6e2a3eb2b6c33c168c2d06f3f8037f34775be724e5bad404a` | `f19dfc6282e50e14cb4d6b5f713aa311f7d31ec297c2b638b96f59e782134ea9` |
| `contrastive_examples:medium` | `retrieval` | `generated/style_research/style_transfer_experiments/method_registry/methods/contrastive_examples.v1.json` | `fadcce5d35184b90ee270885db2de87a81e3f31b4d780f1f85ab8d59fe9455b0` | `760afdb20b72e8eddfb754214ecbe6cff727614c787a0d294058e51db967945e` |
| `llm_close_reading_style_definition:medium` | `llm_close_reading` | `generated/style_research/style_transfer_experiments/method_registry/methods/llm_close_reading_style_definition.v1.json` | `5638eb464e5e718a306208fa410ebcaf8dba006e78602975f282f2da6785c0b9` | `65bea4a90f25dcff55eb18af1e808e4f29c1820154ca0be21c4eaf79ff784f51` |
| `llm_close_reading_cards_statistical_gates:medium` | `hybrid` | `generated/style_research/style_transfer_experiments/method_registry/methods/llm_close_reading_cards_statistical_gates.v1.json` | `dc04f93e0785d8812fc1c18ee94c6140de1cd4f01d94aa14cfece3738e71189c` | `42aba44cd43291b15a2d88928735e252d158f00b4546323151336640417afc65` |

All per-method payloads are contained in and bound by `method_assets/style_transfer_payloads.v1.lock.json`; the active immutable asset is `method_assets/style_transfer_payloads.v1/assets.2be41701eabfecced860e3b8eeea2d48f68cffdc4d23a70ed215d3a0fb0589a3.json`. The close-reading methods additionally bind source packet `source_packet.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json` and result `result.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json`.

`self_critique_repair` was registered as a conditional refinement method but was **not tested**. The protocol permits it only after a non-empty provisional shortlist and independent critique; iteration 1 produced no shortlist.

## Stage 5: Evaluation Rules

For each candidate and arm, the evaluator reports:

- **Target margin:** raw target-author hinge decision margin; not a probability.
- **Paired margin lift:** candidate target margin minus its byte-matched neutral margin.
- **Target rank:** rank of 非天夜翔 among 50 authors; lower is better.
- **Target chunk share:** fraction predicted as 非天夜翔 at rank 1.
- **Deterministic hard fidelity:** paragraph order, CJK-length envelope, Latin token preservation, dialogue-turn surface, quote balance, and reference-copy checks.
- **Independent semantic/readability judgment:** English-grounded blind critique, required only after a method survives deterministic screening.

The deterministic `surface_fidelity.v1` gate is intentionally conservative and fully reproducible. It requires exact paragraph IDs/order and nonempty output; exact multisets of numbers, placeholders, and Latin tokens; a CJK-length ratio of 0.35-2.50 per paragraph for source paragraphs of at least 20 CJK characters; a total CJK ratio of 0.55-1.75; preserved dialogue-start status; balanced Chinese quotation marks; and no copied reference sequence of eight or more CJK characters or reference-name leakage. These checks do **not** prove semantic equivalence.

| Fidelity layer | Iteration-1 status | Failure examples |
| --- | --- | --- |
| Deterministic surface gate | executed for all 432 control/method rows | Missing/duplicate output, paragraph mismatch, number/placeholder/Latin mismatch, length envelope, dialogue-start mismatch, unbalanced quotes |
| Deterministic reference-copy gate | executed for all methods with registered evidence | Introduced eight-CJK reference sequence or reference name/place/lore leakage |
| Blind semantic judgment | not executed because no method survived screening | Added/removed event, role/speaker, negation, causality, chronology, modality, or unsupported detail |
| Blind readability judgment | not executed because no method survived screening | High-severity unnaturalness or readability regression against neutral |

The protocol's top-level `hard_fidelity_failures` list is the union used by the complete multi-stage study. It should not be read as saying that semantic/readability judgments were executed during iteration-1 deterministic screening.

A chunk counts as deterministic style success only if all four conditions hold:

```text
target margin >= 0.218174636
target rank <= 5
paired lift > 0
no hard-fidelity failure
```

Screening promotion requires positive mean lift in both arms, deterministic hard-fidelity failure rate no greater than 10% in either arm, no reference-copy failure, and at most three promoted methods. The 36-row screen is descriptive; it cannot itself satisfy the final 80% endpoint.

### How to read the metrics

- **Rank 1 is weaker than calibrated success.** `Target share` only asks whether the target author has the largest of 50 classifier scores. All 50 scores can still be low, so a rank-1 prediction can have a negative margin and fail the `+0.218` gate.
- **Lift is relative; margin is absolute.** Positive lift means a rewrite moved in the desired direction relative to its own neutral control. It is insufficient when the resulting margin remains below the generated-domain threshold.
- **Own- and cross-author arms answer different questions.** Own-author reconstruction can retain author signal through content or translation; cross-author transfer is the stronger test that the method can impose target style on unfamiliar content.
- **Screening results are descriptive.** With 24 and 12 rows per arm, iteration 1 is a method-family filter, not an 80% success claim. Registered Wilson and cluster-bootstrap inference begins only on the unopened 104/48 confirmation rows.

### Registered confirmation and final endpoints

| Stage / arm | Required evidence for success |
| --- | --- |
| Confirmation, own-author (104 rows) | Style-success point estimate >=80%; >=70% in every one of the eight development books; Wilson 95% lower bound >=70%; 10,000-resample book-cluster bootstrap lower bound >=70%; paired book-cluster bootstrap 95% CI for margin lift excludes zero |
| Confirmation, cross-author (48 rows) | Reported separately; style-success point estimate >=80%; Wilson 95% lower bound >=70%; positive paired lift in at least 10 of 12 authors; author-cluster bootstrap lower bound >=70% |
| Semantic noninferiority | No increase in high-severity candidate failures versus neutral; paired discordant counts reported; cluster-bootstrap 95% upper bound for candidate-minus-neutral failure rate <=0 |
| Readability noninferiority | No high-severity readability regression versus neutral under blinded independent judgment |
| Final validation (80 rows) | Same frozen prompt/model/method/threshold; one-time own-author point estimate >=80%; book-cluster bootstrap lower bound >=70% |

The registered final validation is still a **proxy reconstruction study** using synthetic English. A separate, preregistered pilot on actual Eternal Gate English is required before production adoption; it is not part of evaluation protocol v1 and must not be implied by a future proxy pass.

## Stage 6: Results by Method

![Paired margin lift](../../generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/research_report/charts/iteration1_paired_lift.svg)

![Fidelity failure rate](../../generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/research_report/charts/iteration1_fidelity_failures.svg)

![Absolute target margin](../../generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/research_report/charts/iteration1_target_margin.svg)

### Own-author reconstruction (24 rows)

| Method | Mean target margin | Mean lift | Positive-lift rows | Mean target rank | Target share | Fidelity failures | Style success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral_only:none` | -0.416 | +0.000 | 0/24 | 1.17 | 95.8% | 0/24 (0.0%) | 0/24 |
| `generic_author_style_light:light` | -0.351 | +0.066 | 16/24 | 1.17 | 95.8% | 3/24 (12.5%) | 0/24 |
| `generic_author_style_strong:strong` | -0.421 | -0.004 | 11/24 | 1.25 | 83.3% | 7/24 (29.2%) | 0/24 |
| `global_style_cards:medium` | -0.340 | +0.076 | 17/24 | 1.04 | 95.8% | 11/24 (45.8%) | 0/24 |
| `scene_routed_style_cards:medium` | -0.390 | +0.027 | 18/24 | 1.04 | 95.8% | 8/24 (33.3%) | 0/24 |
| `sentence_flow_cards:medium` | -0.380 | +0.036 | 16/24 | 1.12 | 95.8% | 8/24 (33.3%) | 0/24 |
| `function_word_punctuation_dialogue_cards:medium` | -0.349 | +0.067 | 16/24 | 1.04 | 95.8% | 10/24 (41.7%) | 0/24 |
| `retrieved_examples_only:medium` | -0.461 | -0.044 | 10/24 | 1.25 | 87.5% | 5/24 (20.8%) | 0/24 |
| `scene_cards_plus_examples:medium` | -0.399 | +0.017 | 13/24 | 1.17 | 83.3% | 9/24 (37.5%) | 0/24 |
| `contrastive_examples:medium` | -0.437 | -0.020 | 10/24 | 1.29 | 83.3% | 4/24 (16.7%) | 0/24 |
| `llm_close_reading_style_definition:medium` | -0.422 | -0.006 | 12/24 | 1.21 | 91.7% | 13/24 (54.2%) | 0/24 |
| `llm_close_reading_cards_statistical_gates:medium` | -0.478 | -0.062 | 9/24 | 1.42 | 79.2% | 17/24 (70.8%) | 0/24 |

### Cross-author transfer (12 rows)

| Method | Mean target margin | Mean lift | Positive-lift rows | Mean target rank | Target share | Fidelity failures | Style success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral_only:none` | -1.238 | +0.000 | 0/12 | 14.00 | 0.0% | 1/12 (8.3%) | 0/12 |
| `generic_author_style_light:light` | -1.225 | +0.013 | 9/12 | 12.33 | 0.0% | 1/12 (8.3%) | 0/12 |
| `generic_author_style_strong:strong` | -1.205 | +0.033 | 7/12 | 14.00 | 8.3% | 3/12 (25.0%) | 0/12 |
| `global_style_cards:medium` | -1.183 | +0.054 | 10/12 | 11.92 | 0.0% | 1/12 (8.3%) | 0/12 |
| `scene_routed_style_cards:medium` | -1.201 | +0.037 | 7/12 | 12.08 | 0.0% | 3/12 (25.0%) | 0/12 |
| `sentence_flow_cards:medium` | -1.222 | +0.016 | 8/12 | 12.25 | 0.0% | 2/12 (16.7%) | 0/12 |
| `function_word_punctuation_dialogue_cards:medium` | -1.167 | +0.070 | 8/12 | 10.50 | 8.3% | 6/12 (50.0%) | 0/12 |
| `retrieved_examples_only:medium` | -1.257 | -0.019 | 7/12 | 14.00 | 0.0% | 2/12 (16.7%) | 0/12 |
| `scene_cards_plus_examples:medium` | -1.206 | +0.032 | 7/12 | 12.42 | 0.0% | 2/12 (16.7%) | 0/12 |
| `contrastive_examples:medium` | -1.262 | -0.025 | 5/12 | 16.17 | 0.0% | 0/12 (0.0%) | 0/12 |
| `llm_close_reading_style_definition:medium` | -1.281 | -0.044 | 5/12 | 15.92 | 0.0% | 2/12 (16.7%) | 0/12 |
| `llm_close_reading_cards_statistical_gates:medium` | -1.273 | -0.036 | 5/12 | 18.67 | 0.0% | 4/12 (33.3%) | 0/12 |

### Screening gate matrix

`PASS` below means the method met that one registered screening condition; it does not mean the method achieved chunk-level style success.

| Method | Positive mean lift in both arms | Own fidelity <=10% | Cross fidelity <=10% | Copy-free | Joint promotion gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neutral_only:none` | FAIL | PASS | PASS | PASS | FAIL |
| `generic_author_style_light:light` | PASS | FAIL | PASS | PASS | FAIL |
| `generic_author_style_strong:strong` | FAIL | FAIL | FAIL | PASS | FAIL |
| `global_style_cards:medium` | PASS | FAIL | PASS | PASS | FAIL |
| `scene_routed_style_cards:medium` | PASS | FAIL | FAIL | PASS | FAIL |
| `sentence_flow_cards:medium` | PASS | FAIL | FAIL | PASS | FAIL |
| `function_word_punctuation_dialogue_cards:medium` | PASS | FAIL | FAIL | PASS | FAIL |
| `retrieved_examples_only:medium` | FAIL | FAIL | FAIL | PASS | FAIL |
| `scene_cards_plus_examples:medium` | PASS | FAIL | FAIL | PASS | FAIL |
| `contrastive_examples:medium` | FAIL | FAIL | PASS | PASS | FAIL |
| `llm_close_reading_style_definition:medium` | FAIL | FAIL | FAIL | PASS | FAIL |
| `llm_close_reading_cards_statistical_gates:medium` | FAIL | FAIL | FAIL | PASS | FAIL |

### Method-by-method interpretation

#### `neutral_only:none`

The production-compatible neutral Chinese draft is scored unchanged. This is the negative control and the paired baseline for every lift value.

- Own-author: lift +0.000, mean margin -0.416, target share 95.8%, fidelity failures 0/24.
- Cross-author: lift +0.000, mean margin -1.238, target share 0.0%, fidelity failures 1/12.
- Decision: **not promoted**; control, not promotable; own-author lift was not positive; cross-author lift was not positive. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `generic_author_style_light:light`

The model sees the target author name but no style cards or examples. It is told to make only a light, source-supported recast and to skip weak opportunities.

- Own-author: lift +0.066, mean margin -0.351, target share 95.8%, fidelity failures 3/24.
- Cross-author: lift +0.013, mean margin -1.225, target share 0.0%, fidelity failures 1/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `generic_author_style_strong:strong`

The same author-name-only cue is applied assertively as a stress test. This tests whether stronger generic prompting raises style strength at the cost of fidelity.

- Own-author: lift -0.004, mean margin -0.421, target share 83.3%, fidelity failures 7/24.
- Cross-author: lift +0.033, mean margin -1.205, target share 8.3%, fidelity failures 3/12.
- Decision: **not promoted**; own-author lift was not positive; own-author fidelity failures exceeded 10%; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `global_style_cards:medium`

The model receives compact executable rules derived from 29 target-author training books against 49 comparison authors. Rules cover dialogue tags, punctuation, function words, and sentence flow, each with a semantic guardrail.

- Own-author: lift +0.076, mean margin -0.340, target share 95.8%, fidelity failures 11/24.
- Cross-author: lift +0.054, mean margin -1.183, target share 0.0%, fidelity failures 1/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `scene_routed_style_cards:medium`

The neutral draft is routed to scene types such as dialogue, action, reflection, care, travel, or exposition. Only cards registered for the inferred scene are supplied. Routing uses neutral Chinese, never hidden target metadata.

- Own-author: lift +0.027, mean margin -0.390, target share 95.8%, fidelity failures 8/24.
- Cross-author: lift +0.037, mean margin -1.201, target share 0.0%, fidelity failures 3/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `sentence_flow_cards:medium`

The intervention isolates flow: short-sentence share, mean sentence length, comma-linked clauses, and paragraph progression. It tests whether cadence alone carries enough author signal beyond vocabulary.

- Own-author: lift +0.036, mean margin -0.380, target share 95.8%, fidelity failures 8/24.
- Cross-author: lift +0.016, mean margin -1.222, target share 0.0%, fidelity failures 2/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `function_word_punctuation_dialogue_cards:medium`

The intervention isolates lower-content cues: Chinese function-word families, punctuation, plain speech tags, laughter tags, and micro-reactions. Each rule prohibits adding unsupported emotion, motive, causality, or action.

- Own-author: lift +0.067, mean margin -0.349, target share 95.8%, fidelity failures 10/24.
- Cross-author: lift +0.070, mean margin -1.167, target share 8.3%, fidelity failures 6/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `retrieved_examples_only:medium`

For each neutral draft, retrieval returns three entity-masked target-author train chunks of at most 450 CJK characters. The model gets examples but no explicit style definition.

- Own-author: lift -0.044, mean margin -0.461, target share 87.5%, fidelity failures 5/24.
- Cross-author: lift -0.019, mean margin -1.257, target share 0.0%, fidelity failures 2/12.
- Decision: **not promoted**; own-author lift was not positive; own-author fidelity failures exceeded 10%; cross-author lift was not positive; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `scene_cards_plus_examples:medium`

This hybrid combines scene-routed executable cards with three retrieved masked examples. It tests whether explicit constraints and tacit demonstrations are complementary.

- Own-author: lift +0.017, mean margin -0.399, target share 83.3%, fidelity failures 9/24.
- Cross-author: lift +0.032, mean margin -1.206, target share 0.0%, fidelity failures 2/12.
- Decision: **not promoted**; own-author fidelity failures exceeded 10%; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `contrastive_examples:medium`

The model receives three target-author examples and matched hard-negative examples from other authors. It is told to infer only structural differences shared by target examples and absent from negatives, reducing genre/topic imitation.

- Own-author: lift -0.020, mean margin -0.437, target share 83.3%, fidelity failures 4/24.
- Cross-author: lift -0.025, mean margin -1.262, target share 0.0%, fidelity failures 0/12.
- Decision: **not promoted**; own-author lift was not positive; own-author fidelity failures exceeded 10%; cross-author lift was not positive. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `llm_close_reading_style_definition:medium`

A separate close-reading pass synthesizes discourse, grammar, narrator stance, and scene-mechanics observations from masked target and comparison passages. The transfer model receives that qualitative definition without aggregate statistical gates.

- Own-author: lift -0.006, mean margin -0.422, target share 91.7%, fidelity failures 13/24.
- Cross-author: lift -0.044, mean margin -1.281, target share 0.0%, fidelity failures 2/12.
- Decision: **not promoted**; own-author lift was not positive; own-author fidelity failures exceeded 10%; cross-author lift was not positive; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

#### `llm_close_reading_cards_statistical_gates:medium`

This hybrid combines the LLM close-reading definition with corpus-derived cards and requires qualitative claims to agree with measured target-versus-comparison tendencies.

- Own-author: lift -0.062, mean margin -0.478, target share 79.2%, fidelity failures 17/24.
- Cross-author: lift -0.036, mean margin -1.273, target share 0.0%, fidelity failures 4/12.
- Decision: **not promoted**; own-author lift was not positive; own-author fidelity failures exceeded 10%; cross-author lift was not positive; cross-author fidelity failures exceeded 10%. Deterministic style success was 0/24 own-author and 0/12 cross-author.

## Why Every Method Failed

1. **The absolute margin gap remained large.** Neutral mean margins were `-0.416` own-author and `-1.238` cross-author, while success required `+0.218`. The best method means reached only `-0.340` own-author and `-1.167` cross-author.
2. **Positive relative movement was too small.** Six methods had positive mean lift in both arms, but the maximum mean lifts were only `+0.076` own-author and `+0.070` cross-author.
3. **Fidelity failures increased with many structured interventions.** The dominant deterministic flags were dialogue-turn surface mismatches, changed/missing Latin tokens, and unbalanced dialogue quotes. These are surface hard gates; semantic equivalence was not yet judged.
4. **Raw examples did not teach a transformation.** Retrieved-example and contrastive methods supplied target prose, but not aligned neutral-to-author rewrites. Both moved mean style margin backward in both arms.
5. **Stronger generic prompting was not better.** The strong generic prompt changed own-author lift from `+0.066` to `-0.004` and raised own fidelity failures from 12.5% to 29.2%.

An oracle diagnostic selected, for each row, the highest-margin candidate among all 11 methods. Even this post-hoc upper-bound exercise produced 0/24 own-author and 0/12 cross-author threshold successes. Therefore simple reranking of the existing outputs cannot rescue iteration 1.

## Independent Outcome Evaluation

A separate evaluator reviewed the frozen summary, protocol, calibration, and promotion artifacts. Its verdict was **NO-GO for confirmation**.

| Registered condition | Non-control methods passing |
| --- | ---: |
| Positive mean lift in both arms | 6/11 |
| Own-author fidelity failure rate <=10% | 0/11 |
| Cross-author fidelity failure rate <=10% | 3/11 |
| Zero detected copying | 11/11 |
| All conditions jointly | **0/11** |

The closest gate miss was `generic_author_style_light`: lift `+0.0657` own and `+0.0128` cross, with 3/24 own and 1/12 cross fidelity failures. The 10% rule allows at most 2/24 and 1/12; advancing it because it missed by one row would be a post-hoc relaxation. The complete independent audit is in `audits/screening_iteration1_outcome_audit.md`.

Because the shortlist is empty, independent per-output semantic/readability critiques, lighter-intensity refinement, self-critique repair, confirmation, and final validation were correctly not run. This means iteration 1 establishes deterministic screening failure, not a human judgment that every output is semantically bad or stylistically worthless.

## Threats to Validity

1. **Classifier selection exposure.** Multiple classifier families were compared using the book-disjoint `test` report. Those metrics are valid evidence of out-of-book performance, but they are not a selection-blind estimate after iteration.
2. **Final books are only transfer-held-out.** Their future 80 transfer rows have not been sampled or generated, but the four source books contributed chunks to the Stage 1 classifier benchmark. A stronger academic replication should add target books never used in scorer development or evaluation.
3. **Synthetic English domain.** Proxy English is reconstructed from Chinese and QA'd. It approximates the production direction but cannot reproduce translator-specific omissions, errors, or phrasing in the actual Eternal Gate English source.
4. **Small screening sample.** The 24/12 screening arms support method rejection and direction-of-effect diagnostics, not precise success-rate estimation.
5. **Masking is imperfect content control.** `entity_masked_v3` removes concentrated names and topic terms but may leave content cues or introduce regular mask artifacts.
6. **Single model and one sample per cell.** Iteration 1 measures one frozen `gpt-5.4` output per method/sample. It does not estimate generation variance or portability to another model snapshot.
7. **No human-equivalent outcome judgment was reached.** The deterministic screen blocked all methods before blind semantic/readability judging. Surface fidelity failures are not synonymous with semantic failure, and zero recorded judge failures would not be evidence of equivalence.

## Independent Report Methodology Review

A separate Codex reviewer (`019f57d9-2134-7613-96c8-f8ffe8af10d8`) audited researcher-facing data lineage, exact evidence exposure, metric interpretation, leakage controls, and reproducibility. Its initial review requested clearer final-row materialization status, direct sample and method-asset bindings, QA ledger citations, and separation of deterministic versus judged fidelity layers.

After those revisions, the same reviewer reported **no remaining blocking findings** and rated this document **ADEQUATE for an iteration-1 academic screening report**. The review, finding dispositions, hash-scheme verification, and re-review verdict are recorded in `audits/style_transfer_report_methodology_review.md`.

## Iteration 2 Design (Historical Preregistration)

The following design was frozen after iteration 1 and has now been executed. See the [iteration-2 aligned-pairs report](04_transfer_iteration2_aligned_pairs.md) for current results. The main missing method family was **pseudo-parallel target-author transformation evidence**:

```text
target-author train passage
  -> English semantic source
  -> same production neutral Chinese
  -> aligned pair: neutral Chinese -> original target-author Chinese
  -> retrieve aligned pairs for a new neutral input
  -> generate and fidelity-filter candidate rewrites
```

This differs materially from iteration-1 retrieval: the model sees how neutral prose was transformed into target prose, not only unrelated target examples. The design is supported by Styll/STRAP-style neutral-to-target pseudo-parallel training and by Prompt-and-Rerank's explicit separation of style strength, meaning preservation, and fluency.

Preregistered ablations and diagnostics were:

1. aligned pseudo-parallel demonstrations only;
2. aligned demonstrations plus compact corpus cards;
3. aligned demonstrations with multiple candidate generation and registered reranking;
4. if prompting remains far below threshold, a separately registered learned-adapter or policy-optimization study rather than further prompt wording changes.

Iteration 2 used a fresh 36-row screen and retained 116 confirmation rows. Those confirmation rows and the four final target books remain unopened to transfer generation and selection. No method may enter production until a new method passes screening, independent semantic/readability judgment, confirmation with at least 80% success, and one-time final validation.

## Reproduction

Generate this report and its charts:

```bash
uv run python experiments/iteration1/report_style_transfer_experiment.py
```

Validate sample counts, canonical row hashes, isolation, and leakage controls:

```bash
uv run python experiments/iteration1/style_transfer_research.py validate --sample-id development_proxy_v1
```

Re-run the deterministic 12-arm evaluation from frozen outputs:

```bash
uv run python experiments/iteration1/evaluate_style_transfer_methods.py evaluate \
  --run-id proxy_v1_gpt54_20260711_v2 \
  --selection-file generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.screening_v1_ids.json \
  --analysis-lock generated/style_research/style_transfer_experiments/protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json \
  --scorer-mode load \
  --method neutral_only:none \
  --method generic_author_style_light:light \
  --method generic_author_style_strong:strong \
  --method global_style_cards:medium \
  --method scene_routed_style_cards:medium \
  --method sentence_flow_cards:medium \
  --method function_word_punctuation_dialogue_cards:medium \
  --method retrieved_examples_only:medium \
  --method scene_cards_plus_examples:medium \
  --method contrastive_examples:medium \
  --method llm_close_reading_style_definition:medium \
  --method llm_close_reading_cards_statistical_gates:medium \
  --output-dir generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial
```

## Primary Artifacts

- Corpus split: `generated/style_research/corpus/splits.json`
- Chunk report: `generated/style_research/corpus/chunk_report.json`
- Classifier report: `docs/reports/02_authorship_style_meter.md`
- Proxy sample summary: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.summary.json`
- Runner-visible sample manifest: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.runner_manifest.jsonl`
- Evaluator-only allocation: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.evaluator_allocation.jsonl`
- Evaluator-only hidden targets: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.hidden_targets.jsonl`
- Screening IDs: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.screening_v1_ids.json`
- Confirmation IDs: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.confirmation_v1_ids.json`
- Evaluation protocol: `generated/style_research/style_transfer_experiments/protocols/evaluation_protocol.v1.json`
- Active iteration-1 analysis lock: `generated/style_research/style_transfer_experiments/protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json`
- Model and prompt provenance: `generated/style_research/style_transfer_experiments/protocols/model_run_config.v1.json`
- Method registry and config bindings: `generated/style_research/style_transfer_experiments/method_registry/style_methods.v1.json` and `method_registry/methods/`
- Active method payload lock and asset: `generated/style_research/style_transfer_experiments/method_assets/style_transfer_payloads.v1.lock.json` and `method_assets/style_transfer_payloads.v1/assets.2be41701eabfecced860e3b8eeea2d48f68cffdc4d23a70ed215d3a0fb0589a3.json`
- Close-reading source/result: `generated/style_research/style_transfer_experiments/method_assets/close_reading_source_packets/source_packet.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json` and `method_assets/close_reading_results/result.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json`
- English QA/repair ledgers: `generated/style_research/style_transfer_experiments/runs/development_proxy_v1/proxy_v1_gpt54_20260711_v2/ledgers/english_source_qa.jsonl` and `english_source_repair_qa.round_01.jsonl` through `round_06.jsonl`
- Calibration: `generated/style_research/style_transfer_experiments/calibration/style_meter_threshold.v1.json`
- Screening summary: `generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial/evaluation_summary.json`
- Per-method CSV/JSON/graphs: `generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial/methods/`
- Frozen shortlist decision: `generated/style_research/style_transfer_experiments/promotions/screening_v1.provisional_shortlist.v1.json`
- Independent outcome audit: `generated/style_research/style_transfer_experiments/audits/screening_iteration1_outcome_audit.md`
- Independent report-methodology review: `generated/style_research/style_transfer_experiments/audits/style_transfer_report_methodology_review.md`

## Method Sources

- Krishna, Wieting, and Iyyer (2020), STRAP / style transfer as paraphrase generation: https://aclanthology.org/2020.emnlp-main.55/
- Patel, Andrews, and Callison-Burch (2024 revision), Styll low-resource authorship transfer with neutral-to-target in-context pairs: https://arxiv.org/abs/2212.08986
- Suzgun, Melas-Kyriazi, and Jurafsky (2022), Prompt-and-Rerank: https://aclanthology.org/2022.emnlp-main.141/
- Zhu et al. (2023), StoryTrans for Chinese and English long-story author transfer: https://aclanthology.org/2023.acl-long.827/
- Tao et al. (2024/2025), CAT-LLM Chinese style definitions: https://arxiv.org/abs/2401.05707
- Horvitz et al. (2024), TinyStyler authorship embeddings and reranking: https://aclanthology.org/2024.findings-emnlp.781/
- Liu and May (2025), multi-iteration preference optimization with pseudo-parallel data: https://aclanthology.org/2025.naacl-long.135/
