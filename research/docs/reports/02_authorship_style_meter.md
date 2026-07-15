# Author-Style Classifier on the Current Cleaned Corpus

Generated: 2026-07-15T03:31:27-04:00

## Abstract

This retest measures whether author identity remains recoverable after corpus cleanup, canonical Chinese punctuation normalization, cross-book passage decontamination, and train-only global masking. The benchmark uses book-level splits over 50 authors. The selected scorer removes mask spans before extracting exact character 2-4 gram TF-IDF features. The strongest mask-stripped run reaches 90.0% chunk accuracy, 87.5% balanced accuracy, and 98.1% book-majority accuracy.

## Dataset

- Manifest books: 198
- Cleaned books: 198
- Authors: 50
- Usable books at >=50k cleaned CJK characters: 198
- Primary books at >=120k cleaned CJK characters: 195
- Book split counts: {'train': 88, 'dev': 53, 'test': 53, 'proxy_transfer': 4, 'excluded': 0}
- Clean chunks: 87137
- Train-global-masked chunks: 87137
- Diagnostic global-plus-local chunks: 87137
- Punctuation normalization: `canonical_zh_v1`
- Books changed by punctuation normalization: 195
- Cross-book duplicate fingerprints removed: 1362
- Cross-book duplicate lines removed: 4877
- Remaining checked duplicate fingerprints: 0

Every book is assigned wholly to train, development, test, or target-author proxy holdout. No book contributes chunks to more than one split.

## Methods

- Selected features: exact character 2-4 grams extracted only within unmasked spans   after every `某` run is removed; `TfidfVectorizer`, `max_features=80000`,   `min_df=20`, sublinear TF-IDF, and no hash collisions.
- Text normalization: whitespace is removed for n-gram extraction. Dataset cleanup   canonicalizes equivalent punctuation encodings while preserving punctuation roles.
- Classifiers: SGD hinge with and without class balancing; random seed 13.
- Mask fitting: one global content vocabulary is learned from train books only and then   applied identically to every split. It is the only current scorer input view.
- Diagnostic views: clean text and global-plus-book-local v3 are retained to measure   content and preprocessing effects; v3 is not eligible for scorer selection.
- Metrics: chunk accuracy, balanced accuracy, macro F1, target-author precision/recall/F1,   and book-level majority-vote accuracy.

## Main Result

Highest mask-stripped classifier: **SGD linear SVM / Exact character n-grams within unmasked spans only** on `train_global_masked`.

- Chunk accuracy: 90.0%
- Balanced accuracy: 87.5%
- Book-majority accuracy: 98.1%
- Target-author F1: 89.5%

Selected provisional authorship proxy: **SGD linear SVM (unweighted) / Exact character n-grams within unmasked spans only** on `train_global_masked`.

- Chunk accuracy: 89.8%
- Balanced accuracy: 87.9%
- Macro F1: 87.4%
- Book-majority accuracy: 96.2%
- Target-author precision: 80.3%
- Target-author F1: 89.1%
- Target-author recall: 100.0%
- Majority baseline: 5.9%
- Features: 80000

![Train-global-masked chunk accuracy](../../generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/report_charts/masked_chunk_accuracy.svg)

![Train-global-masked balanced accuracy](../../generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/report_charts/masked_balanced_accuracy.svg)

## All Current Runs

| Rank | View | Classifier | Features | Chunk acc. | Balanced acc. | Book acc. | Target F1 | Target recall | Majority baseline |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | train_global_masked | SGD linear SVM | Exact character n-grams within unmasked spans only | 90.0% | 87.5% | 98.1% | 89.5% | 100.0% | 5.9% |
| 2 | train_global_masked | SGD linear SVM (unweighted) | Exact character n-grams within unmasked spans only | 89.8% | 87.9% | 96.2% | 89.1% | 100.0% | 5.9% |
| 3 | entity_masked_v3 | SGD linear SVM (unweighted) | Character n-grams | 89.4% | 86.1% | 98.1% | 88.5% | 100.0% | 5.9% |
| 4 | entity_masked_v3 | SGD linear SVM | Character n-grams | 89.1% | 85.7% | 100.0% | 90.0% | 100.0% | 5.9% |
| 5 | entity_masked_v3 | SGD linear SVM (unweighted) | Exact character n-grams within unmasked spans only | 89.0% | 85.5% | 98.1% | 83.9% | 100.0% | 5.9% |
| 6 | entity_masked_v3 | SGD linear SVM | Exact character n-grams within unmasked spans only | 88.8% | 84.8% | 100.0% | 87.2% | 100.0% | 5.9% |
| 7 | train_global_masked | SGD linear SVM (unweighted) | Character n-grams | 86.6% | 85.0% | 96.2% | 92.0% | 100.0% | 5.9% |
| 8 | train_global_masked | SGD linear SVM | Character n-grams | 85.5% | 83.3% | 92.5% | 90.8% | 100.0% | 5.9% |
| 9 | clean | SGD linear SVM | Character n-grams | 72.1% | 63.7% | 79.2% | 57.7% | 100.0% | 5.9% |
| 10 | clean | SGD linear SVM (unweighted) | Character n-grams | 71.1% | 62.7% | 81.1% | 55.8% | 100.0% | 5.9% |

## Masked vs Clean

Global masking removes many book-specific names and setting terms. Performance must therefore be interpreted as content-resistant author signal, not as proof that every learned feature is literary style.

| Classifier | Features | Clean chunk acc. | Masked chunk acc. | Clean - masked | Masked balanced acc. |
|---|---|---:|---:|---:|---:|
| SGD linear SVM (unweighted) | Character n-grams | 71.1% | 86.6% | -15.4pp | 85.0% |
| SGD linear SVM | Character n-grams | 72.1% | 85.5% | -13.4pp | 83.3% |

![Clean to masked gap](../../generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/report_charts/clean_to_masked_gap.svg)

## Masking Ablation

The parent benchmark compares the fixed train-global vocabulary with the ineligible book-local supplement. Separate collapsed-run, topology-only, and mask-stripped controls then test whether mask artifacts explain the retained author signal.

| Classifier | Features | Train-global only | Global + label-blind local | Combined - global |
|---|---|---:|---:|---:|
| SGD linear SVM | Character n-grams | 85.5% | 89.1% | +3.6pp |
| SGD linear SVM | Exact character n-grams within unmasked spans only | 90.0% | 88.8% | -1.2pp |
| SGD linear SVM (unweighted) | Character n-grams | 86.6% | 89.4% | +2.9pp |
| SGD linear SVM (unweighted) | Exact character n-grams within unmasked spans only | 89.8% | 89.0% | -0.7pp |

The final mask-stripped global run improves over the original global run by +3.20pp test accuracy. Its development accuracy is 87.94% and development balanced accuracy is 84.57%. All registered v1 and v2 mask-artifact checks pass.

## Per-Author Results

All 50 authors under the recommended method, ordered from lowest to highest recall:

| Author | Precision | Recall | F1 | Support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 青色羽翼 | 95.9% | 25.1% | 39.8% | 837 | 210 | 9 | 627 |
| 语笑阑珊 | 93.8% | 25.5% | 40.1% | 475 | 121 | 8 | 354 |
| 李温酒 | 98.2% | 52.1% | 68.1% | 1490 | 776 | 14 | 714 |
| 龙柒 | 91.7% | 60.3% | 72.8% | 638 | 385 | 35 | 253 |
| 多金少女猫 | 99.3% | 65.4% | 78.8% | 459 | 300 | 2 | 159 |
| 红口白牙 | 97.9% | 65.6% | 78.6% | 288 | 189 | 4 | 99 |
| 北南 | 87.4% | 67.3% | 76.0% | 257 | 173 | 25 | 84 |
| 一十四洲 | 77.9% | 70.9% | 74.3% | 368 | 261 | 74 | 107 |
| 唐酒卿 | 94.5% | 71.9% | 81.7% | 616 | 443 | 26 | 173 |
| 引路星 | 96.8% | 76.5% | 85.5% | 200 | 153 | 5 | 47 |
| 酱子贝 | 90.1% | 80.3% | 84.9% | 238 | 191 | 21 | 47 |
| 吾九殿 | 99.7% | 82.9% | 90.5% | 468 | 388 | 1 | 80 |
| 骑鲸南去 | 98.6% | 84.3% | 90.9% | 833 | 702 | 10 | 131 |
| 微风几许 | 87.9% | 85.4% | 86.6% | 205 | 175 | 24 | 30 |
| 公子于歌 | 95.3% | 88.8% | 92.0% | 457 | 406 | 20 | 51 |
| 妾在山阳 | 98.1% | 89.8% | 93.8% | 937 | 841 | 16 | 96 |
| 小霄 | 100.0% | 90.4% | 94.9% | 374 | 338 | 0 | 36 |
| 若星若辰 | 97.7% | 90.4% | 93.9% | 426 | 385 | 9 | 41 |
| 吕天逸 | 81.9% | 90.4% | 86.0% | 230 | 208 | 46 | 22 |
| 七流 | 46.0% | 90.5% | 61.0% | 598 | 541 | 634 | 57 |
| 莫晨欢 | 97.4% | 90.6% | 93.9% | 770 | 698 | 19 | 72 |
| 绿野千鹤 | 76.1% | 92.1% | 83.3% | 266 | 245 | 77 | 21 |
| 漫漫何其多 | 97.8% | 94.4% | 96.1% | 285 | 269 | 6 | 16 |
| 蝶之灵 | 65.6% | 95.4% | 77.8% | 1405 | 1341 | 703 | 64 |
| 墨香铜臭 | 93.1% | 96.1% | 94.6% | 635 | 610 | 45 | 25 |
| 望三山 | 98.0% | 96.2% | 97.1% | 554 | 533 | 11 | 21 |
| black_di | 89.2% | 96.3% | 92.6% | 189 | 182 | 22 | 7 |
| 西西特 | 99.7% | 96.7% | 98.2% | 2217 | 2143 | 6 | 74 |
| 颜凉雨 | 98.3% | 97.1% | 97.7% | 730 | 709 | 12 | 21 |
| 衣落成火 | 95.3% | 97.8% | 96.5% | 2349 | 2297 | 113 | 52 |
| 春风遥 | 94.8% | 98.1% | 96.4% | 672 | 659 | 36 | 13 |
| 比卡比 | 62.9% | 98.1% | 76.7% | 320 | 314 | 185 | 6 |
| 碉堡堡 | 98.8% | 98.3% | 98.5% | 860 | 845 | 10 | 15 |
| 青衣杏林 | 99.3% | 98.3% | 98.8% | 2759 | 2712 | 18 | 47 |
| 拉棉花糖的兔子 | 67.2% | 98.4% | 79.9% | 383 | 377 | 184 | 6 |
| 梦溪石 | 81.3% | 98.8% | 89.2% | 589 | 582 | 134 | 7 |
| 马户子君 | 95.5% | 99.6% | 97.5% | 234 | 233 | 11 | 1 |
| 坏猫霸霸 | 93.3% | 99.6% | 96.4% | 545 | 543 | 39 | 2 |
| 木瓜黄 | 81.9% | 99.7% | 89.9% | 332 | 331 | 73 | 1 |
| 桑沃 | 98.8% | 99.7% | 99.3% | 2381 | 2375 | 28 | 6 |
| 西子绪 | 93.3% | 99.8% | 96.4% | 490 | 489 | 35 | 1 |
| 稚楚 | 83.7% | 99.8% | 91.0% | 519 | 518 | 101 | 1 |
| 非天夜翔 | 80.3% | 100.0% | 89.1% | 2134 | 2134 | 524 | 0 |
| 风流书呆 | 90.9% | 100.0% | 95.2% | 1144 | 1144 | 115 | 0 |
| 妄鸦 | 91.1% | 100.0% | 95.3% | 561 | 561 | 55 | 0 |
| 墨西柯 | 92.2% | 100.0% | 95.9% | 341 | 341 | 29 | 0 |
| 淮上 | 92.2% | 100.0% | 95.9% | 506 | 506 | 43 | 0 |
| priest | 93.2% | 100.0% | 96.5% | 481 | 481 | 35 | 0 |
| 木苏里 | 95.2% | 100.0% | 97.6% | 439 | 439 | 22 | 0 |
| 巫哲 | 97.5% | 100.0% | 98.7% | 510 | 510 | 13 | 0 |

![Per-author recall](../../generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/report_charts/best_method_per_author_recall.svg)

## Top Confusions

| Gold author | Predicted author | Count |
|---|---|---:|
| 李温酒 | 蝶之灵 | 350 |
| 李温酒 | 七流 | 296 |
| 青色羽翼 | 七流 | 262 |
| 龙柒 | 蝶之灵 | 245 |
| 语笑阑珊 | 比卡比 | 134 |
| 青色羽翼 | 非天夜翔 | 124 |
| 唐酒卿 | 非天夜翔 | 107 |
| 语笑阑珊 | 梦溪石 | 104 |
| 北南 | 非天夜翔 | 73 |
| 多金少女猫 | 衣落成火 | 55 |

## Interpretation

- Use masked balanced accuracy and target-author F1 together; raw accuracy alone is   insufficient under uneven book and chunk counts.
- Treat book-majority accuracy as a sanity check because most authors contribute only one   test book.
- Character n-grams remain leakage-sensitive. Train-only masking, punctuation normalization,   exact decontamination, and mask-stripped extraction control known shortcuts but cannot   remove every topic, formatting, or source artifact.
- This classifier is currently an authorship proxy, not a validated continuous style   meter. Generated-text calibration, uncertainty estimates, semantic fidelity, and   Chinese-readability evaluation remain separate requirements.

## Reproduction

```bash
uv run author-style-research authorship-supervised \
  --views clean,train_global_masked,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods char_ngrams \
  --classifiers sgd_hinge,sgd_hinge_unbalanced \
  --max-char-features 80000 \
  --char-min-df 20 \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_cleaned \
  --jobs -1 --max-iter 3000

uv run author-style-research mask-artifact-audit --jobs -1

uv run author-style-research authorship-report \
  --baseline-evaluation-file generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/evaluator_review.md \
  --evaluation-file generated/style_research/benchmarks/mask_artifact_ablation_v2/evaluator_review.md
```

## Independent Evaluation

### Parent Benchmark Audit

<!-- benchmark-result-sha256: 25d9d966bbd7987279693a18dfe4b75e3da274f991a3a092c13f5db2f3c9d727 -->
<!-- evaluator-model: gpt-5.6-terra -->

## Material Passport

- Verification Status: ANALYZED
- Scope: code, tests, aggregate JSON, provenance, QA statistics, and confusion/per-author metrics only. No raw novel, cleaned full-book, or chunk text was inspected.
- Result SHA-256 independently matched the supplied value.
- The recorded benchmark-script and mask-plan hashes match the current files. The result binds 87,137 rows in each of the three evaluated views.

## Independent Evaluation

### Decision

The prior global-mask leakage is fixed for the stated threat: global mask terms are fitted on the 88 training books only; held-out corpus statistics are disallowed; the transform does not use author labels. This is enforced both in code and in focused tests.

Exact cross-book decontamination is a meaningful and adequate control for direct, sufficiently long exact passage reuse under its declared rule. It is not a complete control for near duplicates, shorter repeats, paraphrase, or source/formatting confounds.

The defensible provisional authorship proxy is the **unweighted SGD hinge classifier on `train_global_masked`**, not `entity_masked_v3`. This follows development-set selection and construct validity: it has the better development accuracy (86.27% vs. 85.83%), higher target F1 (92.00% vs. 88.46% on test), and does not adapt its term vocabulary to each evaluated book. The current test set must now be treated as spent for model/view choice.

The >80% milestone is met for a bounded, held-out-book, 50-author attribution proxy. It is not sufficient to claim a continuous author-style meter for generated translations.

### Evidence

The benchmark contains 198 books from 50 authors: 88 train, 53 development, 53 test, and 4 target-author proxy-transfer books. Class imbalance is substantial: training support ranges from 78 to 5,465 chunks per author (about 70:1).

| View | Model | Dev accuracy / balanced accuracy / target F1 | Test accuracy / balanced accuracy / target F1 |
|---|---|---:|---:|
| Clean | Class-balanced | 78.11 / 68.19 / 64.38% | 72.13 / 63.67 / 57.65% |
| Clean | Unweighted | 76.72 / 66.95 / 62.05% | 71.12 / 62.66 / 55.78% |
| Train-global masked | Class-balanced | 85.17 / 81.57 / 90.73% | 85.50 / 83.32 / 90.79% |
| Train-global masked | Unweighted | **86.27 / 82.57 / 92.00%** | **86.56 / 85.01 / 92.00%** |
| Global + local v3 | Class-balanced | 86.05 / 83.11 / 90.59% | 89.06 / 85.69 / 89.99% |
| Global + local v3 | Unweighted | 85.83 / 82.48 / 89.75% | 89.43 / 86.13 / 88.46% |

The clean-to-global gain is very large: +15.44 percentage points in unweighted test accuracy and +22.35 points in balanced accuracy. This supports the conclusion that clean character n-grams contain strong nuisance/topic/source signals; it does not, by itself, establish that the retained signal is literary style.

`entity_masked_v3` adds label-blind, book-local masking. Its unweighted test accuracy rises 2.87 points over global-only, but its development accuracy falls 0.44 points and its target F1 falls 3.54 points. More importantly, it preserves mask length and placement: QA statistics show 24.26 million generic mask characters (18.87% of CJK characters), versus 4.31 million (3.36%) in the global-only view. Character n-grams can learn this masking topology, density, and run-length structure.

Decontamination removed 1,362 exact fingerprints, 4,877 lines, and 222,241 CJK characters across 28 books; zero fingerprints remained under the declared 1-line/80-CJK, 2-line/100-CJK, and 3-line/120-CJK rules. This directly addresses long exact copied passages across books.

Per-author performance remains uneven. Under the recommended global-only unweighted view, the weakest recalls are 青色羽翼 8.60% (F1 15.82%), 李温酒 35.17% (51.60%), 语笑阑珊 40.42% (54.78%), and 龙柒 41.69% (58.72%). Under v3, several remain weak, including 青色羽翼 23.89%, 多金少女猫 23.97%, and 语笑阑珊 34.74%. Aggregate accuracy therefore masks material author-specific failure.

### Threats to Validity

- The v3 local vocabulary is label-blind but is still computed from each test/development book. It is transductive, input-adaptive preprocessing rather than a fixed scorer input contract.
- Exact decontamination misses paraphrase, partial overlaps below thresholds, segmentation-dependent matches, and shared source/platform/genre artifacts. It also uses all books before splitting; label-free, but still a cross-split corpus transform.
- The test result contains 53 books, generally one book per non-target author. Book-majority results are therefore coarse: global-only unweighted is 51/53 (96.23%) and v3 unweighted 52/53 (98.11%). Approximate unclustered Wilson intervals are wide—about 87.2–99.0% and 90.0–99.6%, respectively—and chunk-level intervals would be anti-conservative because chunks within books are correlated.
- The test set is exposed across six view/model comparisons and in report ranking. The hard-coded current v3 run makes present reporting deterministic, but does not demonstrate that the choice was registered before these test results were observed.
- Target recall is 100%, but global-only has 371 target false positives and v3 has 557. A target-author F1 alone does not establish a usable deployment precision at the intended prevalence.
- The benchmark evaluates discrete attribution only. It records hard predictions, not calibrated uncertainty or a monotonic score of “degree of target style.” Generated translations introduce domain shift, semantic-fidelity variation, and translationese that are not represented by this task.

### Fallacy Scan

Coverage: **11/11 statistical fallacy checks completed.**

| Check | Status | Assessment |
|---|---|---|
| Simpson’s paradox | CAUTION | Aggregate accuracy obscures extreme per-author failures; no reversal analysis or hierarchical estimate is reported. |
| Ecological fallacy | CAUTION | Group-level attribution performance cannot establish style intensity for an individual generated translation. |
| Berkson’s paradox | CAUTION | Eligibility requires sufficiently long, available books and at least three books per author; selection effects are plausible. |
| Collider bias | CAUTION | Conditioning on cleaned-corpus eligibility and source-quality filters may induce associations between author, source, and retained text properties. |
| Base-rate neglect | CAUTION | Balanced accuracy and majority baseline are reported, but deployment prevalence and calibrated target PPV are not. |
| Regression to the mean | NOT DETECTED | No pre/post extreme-group intervention design is used. |
| Survivorship bias | CAUTION | The selected 50 authors/books are a survivable, well-represented corpus, not a population sample of authors or translations. |
| Look-elsewhere effect | CAUTION | Six test-facing comparisons are reported; no multiplicity control or held-out model-selection tier is documented. |
| Garden of forking paths | CAUTION | Numerous cleaning, masking, and feature choices exist. No immutable time-stamped preregistration artifact was supplied for this audit. |
| Correlation is not causation | CAUTION | Better masked attribution cannot be interpreted as masking causing measurement of literary style, nor as generated text becoming more author-like. |
| Reverse causality | NOT DETECTED | No directional causal claim is required for the descriptive benchmark; avoid introducing one in interpretation. |

### Required Next Actions

1. **Yes—require a mask-token-stripped or collapsed-mask ablation before finalizing the scorer.** Run it for global-only and v3, ideally alongside a length-matched random-mask control. Report mask density/run-length-only predictive performance.
2. Freeze `train_global_masked` plus unweighted SGD hinge as the provisional proxy using the development result, with a versioned selection manifest. Do not select on the current test again.
3. Treat the current test set as consumed; create a fresh final holdout or use repeated nested, book-stratified cross-validation for subsequent selection.
4. Report book-cluster bootstrap confidence intervals, multiple seeds, per-author intervals, macro metrics, and full confusion matrices—not only aggregate accuracy.
5. Extend decontamination to fuzzy/near-duplicate and source-provenance checks, while retaining the present exact-passages audit.
6. Validate a continuous meter separately: calibrated margins/probabilities, generated-translation evaluation, semantic-fidelity and readability gates, and blinded human judgments with agreement and monotonicity tests.
7. Investigate the weakest authors and recurrent confusions before using any aggregate score as a broad authorship claim.

### Reproducibility

The completed result is hash-bound to the supplied SHA-256 and records its mask-plan, benchmark-script, and chunk-file digests. Current benchmark-script and mask-plan digests match those bindings. I did not rerun training or inspect text-bearing inputs; accordingly, this is an analysis audit, not independent execution verification.

### Final Mask-Artifact Audit

<!-- ablation-result-sha256: 6ea4253a30544414ea03926fbdaf47e0e05ce4accc68a21d4b5b18a2fea8763a -->
<!-- evaluator-model: gpt-5.6-terra; reasoning: high; service-tier: priority; independent: true -->

## Audit decision

**Pass.** The registered `train_global_masked` unweighted mask-stripped baseline passes every gate condition: dev accuracy/balanced accuracy exceed 80%, test balanced accuracy exceeds 80%, accuracy changes are improvements (−1.67pp dev; −3.20pp test), and the required prior controls pass.

**No further masking iteration is needed** for the bounded held-out-book attribution claim. The representation excludes mask tokens, mask-run/topology features, and cross-mask n-grams. This does not establish that all residual lexical-survival effects are absent.

**Still invalid:** claims that predictions or margins measure a continuous degree of generated-text style; claims of calibration or monotonic style scoring; and claims of validity under generated-text domain shift, translationese, semantic-fidelity variation, or readability variation.

## Conclusion

Freeze **SGD linear SVM (unweighted) / Exact character n-grams within unmasked spans only** on `train_global_masked` as the versioned authorship-attribution proxy for generated-domain calibration. Its test metrics are descriptive, not a basis for choosing between classifiers. Do not promote it to a continuous style meter until generated-text and human-rating calibration are complete.
