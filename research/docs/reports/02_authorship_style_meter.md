# Author-Style Classifier Retest on 50-Author BL Corpus

Generated: 2026-07-10

## Abstract

This retest evaluates whether author identity remains recoverable after entity masking and whether the classifier is strong enough to serve as a proxy style meter for later author-style transfer experiments. The strongest run is an exact character n-gram TF-IDF model with SGD hinge loss on `entity_masked_v3`, reaching 88.2% chunk accuracy, 84.1% balanced accuracy, and 94.3% book-majority accuracy over 50 authors. For target-author style-transfer scoring, the recommended meter is the class-balanced exact n-gram hinge model, which trades a small amount of overall accuracy for higher target-author F1.

## Dataset

- Manifest books: 200
- Cleaned books: 200
- Authors: 50
- Usable books at >=50k cleaned CJK characters: 199
- Primary books at >=120k cleaned CJK characters: 196
- Book split counts: {'train': 89, 'dev': 53, 'test': 53, 'proxy_transfer': 4, 'excluded': 1}
- Clean chunks: 87174
- `entity_masked_v3` chunks: 87174

The split is book-level: chunks from a given book remain in one split. This is stricter than random chunk splitting and is intended to reduce within-book leakage.

## Methods

- Exact character n-grams: sklearn `TfidfVectorizer`, character 2-4 grams, `max_features=80000`, `min_df=20`, sublinear TF-IDF, real vocabulary, no hash collisions.
- Hashed character n-grams: sklearn `HashingVectorizer`, character 2-4 grams, 262144 hash buckets, `alternate_sign=False`, TF-IDF normalization.
- Interpretable baselines: punctuation/dialogue, sentence/paragraph length, function-character, richer Chinese function-word lexicon, function words plus characters, and combined interpretable feature families.
- Classifiers: SGD hinge, unweighted SGD hinge, SGD logistic, unweighted SGD logistic, passive-aggressive, and ComplementNB where applicable.
- Metrics: chunk-level accuracy, balanced accuracy, macro F1, target-author F1/recall, majority-class baseline, and book-level majority-vote accuracy.

## Main Result

Highest overall classifier: **SGD linear SVM (unweighted) / Character n-grams** on `entity_masked_v3`.

- Chunk accuracy: 88.2%
- Balanced accuracy: 84.1%
- Book-majority accuracy: 94.3%
- Target-author F1: 85.2%

Recommended target-author proxy meter: **SGD linear SVM / Character n-grams** on `entity_masked_v3`.

- Chunk accuracy: 87.8%
- Balanced accuracy: 83.6%
- Macro F1: 82.3%
- Book-majority accuracy: 90.6%
- Target-author F1: 85.9%
- Target-author recall: 100.0%
- Majority baseline: 6.0%
- Features: 80000

![Masked chunk accuracy](../../generated/style_research/benchmarks/author_classifier_retest_50authors_report/charts/masked_chunk_accuracy.svg)

![Masked balanced accuracy](../../generated/style_research/benchmarks/author_classifier_retest_50authors_report/charts/masked_balanced_accuracy.svg)

## Masked vs Clean

The strongest n-gram models score higher on masked text than clean text. This is surprising but plausible: entity masking removes book-specific names and topical anchors that can make train/test books from the same author less distributionally consistent. It does not by itself prove the model is purely stylistic, but it does show author signal survives content masking strongly.

| Classifier | Features | Clean chunk acc. | Masked chunk acc. | Clean - masked | Masked balanced acc. |
|---|---|---:|---:|---:|---:|
| SGD linear SVM (unweighted) | Character n-grams | 71.4% | 88.2% | -16.8pp | 84.1% |
| SGD linear SVM | Character n-grams | 70.7% | 87.8% | -17.1pp | 83.6% |
| SGD linear SVM | Character n-grams (hashed) | 64.1% | 85.0% | -20.9pp | 80.5% |
| SGD linear SVM (unweighted) | Character n-grams (hashed) | 63.7% | 84.0% | -20.3pp | 78.7% |
| Passive-aggressive linear classifier | Character n-grams (hashed) | 55.6% | 81.7% | -26.1pp | 76.4% |
| Passive-aggressive linear classifier | Combined + function words | 69.7% | 69.5% | +0.3pp | 63.9% |
| SGD logistic regression | Character n-grams (hashed) | 33.9% | 67.8% | -33.8pp | 58.8% |
| SGD linear SVM | Combined + function words | 67.6% | 67.2% | +0.4pp | 62.9% |
| SGD linear SVM (unweighted) | Combined + function words | 67.3% | 65.4% | +1.9pp | 61.5% |
| Passive-aggressive linear classifier | Function words + characters | 60.0% | 60.0% | +0.0pp | 53.9% |
| SGD linear SVM (unweighted) | Function words + characters | 57.5% | 59.7% | -2.1pp | 52.5% |
| SGD logistic regression | Combined + function words | 59.7% | 59.3% | +0.4pp | 58.5% |
| SGD linear SVM | Function words + characters | 60.4% | 58.3% | +2.1pp | 54.2% |
| Passive-aggressive linear classifier | Combined interpretable | 56.3% | 57.7% | -1.5pp | 53.4% |
| SGD linear SVM (unweighted) | Combined interpretable | 56.0% | 55.0% | +1.0pp | 49.5% |
| SGD logistic regression | Function words + characters | 55.4% | 54.9% | +0.5pp | 51.2% |
| SGD linear SVM | Combined interpretable | 54.3% | 54.9% | -0.6pp | 52.7% |
| SGD logistic regression (unweighted) | Character n-grams (hashed) | 27.0% | 51.3% | -24.2pp | 39.6% |
| SGD logistic regression | Combined interpretable | 50.0% | 50.0% | -0.0pp | 49.1% |
| SGD linear SVM (unweighted) | Function-character | 45.7% | 45.0% | +0.7pp | 35.4% |
| Complement Naive Bayes | Character n-grams (hashed) | 42.5% | 44.1% | -1.6pp | 27.5% |
| Passive-aggressive linear classifier | Function-character | 43.1% | 42.6% | +0.5pp | 36.5% |
| SGD linear SVM | Function-character | 44.5% | 42.5% | +2.0pp | 37.0% |
| SGD logistic regression | Function-character | 39.7% | 40.6% | -0.9pp | 35.2% |
| SGD logistic regression | Chinese function words | 40.4% | 39.4% | +1.0pp | 37.8% |
| SGD linear SVM (unweighted) | Chinese function words | 41.0% | 38.9% | +2.0pp | 34.3% |
| SGD linear SVM | Chinese function words | 39.7% | 38.5% | +1.2pp | 36.4% |
| Passive-aggressive linear classifier | Chinese function words | 38.2% | 38.0% | +0.2pp | 34.9% |
| Complement Naive Bayes | Function words + characters | 25.4% | 25.2% | +0.2pp | 14.4% |
| Complement Naive Bayes | Combined + function words | 24.5% | 24.4% | +0.1pp | 15.7% |
| Complement Naive Bayes | Chinese function words | 23.9% | 23.6% | +0.2pp | 13.8% |
| SGD logistic regression | Punctuation/dialogue | 21.5% | 21.5% | -0.0pp | 21.9% |
| Complement Naive Bayes | Function-character | 19.1% | 19.2% | -0.1pp | 8.9% |
| Complement Naive Bayes | Combined interpretable | 17.4% | 17.4% | -0.0pp | 10.7% |
| SGD logistic regression | Sentence/paragraph length | 16.7% | 16.7% | +0.0pp | 15.3% |
| Complement Naive Bayes | Punctuation/dialogue | 13.9% | 14.0% | -0.1pp | 7.9% |
| Complement Naive Bayes | Sentence/paragraph length | 7.9% | 7.9% | +0.0pp | 4.1% |

![Clean to masked gap](../../generated/style_research/benchmarks/author_classifier_retest_50authors_report/charts/clean_to_masked_gap.svg)

## Method Comparison

| Rank | View | Classifier | Features | Chunk acc. | Balanced acc. | Book acc. | Target F1 | Target recall | Majority baseline |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | entity_masked_v3 | SGD linear SVM (unweighted) | Character n-grams | 88.2% | 84.1% | 94.3% | 85.2% | 100.0% | 6.0% |
| 2 | entity_masked_v3 | SGD linear SVM | Character n-grams | 87.8% | 83.6% | 90.6% | 85.9% | 100.0% | 6.0% |
| 3 | entity_masked_v3 | SGD linear SVM | Character n-grams (hashed) | 85.0% | 80.5% | 92.5% | 77.5% | 100.0% | 6.0% |
| 4 | entity_masked_v3 | SGD linear SVM (unweighted) | Character n-grams (hashed) | 84.0% | 78.7% | 90.6% | 70.7% | 100.0% | 6.0% |
| 5 | entity_masked_v3 | Passive-aggressive linear classifier | Character n-grams (hashed) | 81.7% | 76.4% | 86.8% | 67.8% | 100.0% | 6.0% |
| 6 | entity_masked_v3 | Passive-aggressive linear classifier | Combined + function words | 69.5% | 63.9% | 92.5% | 78.7% | 98.7% | 6.0% |
| 7 | entity_masked_v3 | SGD logistic regression | Character n-grams (hashed) | 67.8% | 58.8% | 69.8% | 53.5% | 99.7% | 6.0% |
| 8 | entity_masked_v3 | SGD linear SVM | Combined + function words | 67.2% | 62.9% | 92.5% | 79.3% | 96.4% | 6.0% |
| 9 | entity_masked_v3 | SGD linear SVM (unweighted) | Combined + function words | 65.4% | 61.5% | 92.5% | 78.1% | 98.2% | 6.0% |
| 10 | entity_masked_v3 | Passive-aggressive linear classifier | Function words + characters | 60.0% | 53.9% | 90.6% | 84.3% | 90.1% | 6.0% |
| 11 | entity_masked_v3 | SGD linear SVM (unweighted) | Function words + characters | 59.7% | 52.5% | 88.7% | 69.1% | 98.4% | 6.0% |
| 12 | entity_masked_v3 | SGD logistic regression | Combined + function words | 59.3% | 58.5% | 88.7% | 78.5% | 93.3% | 6.0% |
| 13 | entity_masked_v3 | SGD linear SVM | Function words + characters | 58.3% | 54.2% | 96.2% | 83.0% | 91.5% | 6.0% |
| 14 | entity_masked_v3 | Passive-aggressive linear classifier | Combined interpretable | 57.7% | 53.4% | 84.9% | 75.1% | 92.0% | 6.0% |
| 15 | entity_masked_v3 | SGD linear SVM (unweighted) | Combined interpretable | 55.0% | 49.5% | 79.2% | 62.4% | 96.8% | 6.0% |
| 16 | entity_masked_v3 | SGD logistic regression | Function words + characters | 54.9% | 51.2% | 90.6% | 81.0% | 91.1% | 6.0% |
| 17 | entity_masked_v3 | SGD linear SVM | Combined interpretable | 54.9% | 52.7% | 81.1% | 74.4% | 89.4% | 6.0% |
| 18 | entity_masked_v3 | SGD logistic regression (unweighted) | Character n-grams (hashed) | 51.3% | 39.6% | 49.1% | 26.9% | 100.0% | 6.0% |
| 19 | entity_masked_v3 | SGD logistic regression | Combined interpretable | 50.0% | 49.1% | 81.1% | 73.9% | 87.3% | 6.0% |
| 20 | entity_masked_v3 | SGD linear SVM (unweighted) | Function-character | 45.0% | 35.4% | 69.8% | 52.8% | 95.9% | 6.0% |
| 21 | entity_masked_v3 | Complement Naive Bayes | Character n-grams (hashed) | 44.1% | 27.5% | 30.2% | 53.1% | 99.8% | 6.0% |
| 22 | entity_masked_v3 | Passive-aggressive linear classifier | Function-character | 42.6% | 36.5% | 71.7% | 68.0% | 85.4% | 6.0% |
| 23 | entity_masked_v3 | SGD linear SVM | Function-character | 42.5% | 37.0% | 77.4% | 72.7% | 80.4% | 6.0% |
| 24 | entity_masked_v3 | SGD logistic regression | Function-character | 40.6% | 35.2% | 69.8% | 70.7% | 80.0% | 6.0% |
| 25 | entity_masked_v3 | SGD logistic regression | Chinese function words | 39.4% | 37.8% | 94.3% | 63.4% | 66.4% | 6.0% |
| 26 | entity_masked_v3 | SGD linear SVM (unweighted) | Chinese function words | 38.9% | 34.3% | 77.4% | 53.4% | 86.8% | 6.0% |
| 27 | entity_masked_v3 | SGD linear SVM | Chinese function words | 38.5% | 36.4% | 94.3% | 60.8% | 66.2% | 6.0% |
| 28 | entity_masked_v3 | Passive-aggressive linear classifier | Chinese function words | 38.0% | 34.9% | 86.8% | 58.8% | 62.2% | 6.0% |
| 29 | entity_masked_v3 | Complement Naive Bayes | Function words + characters | 25.2% | 14.4% | 22.6% | 38.8% | 92.7% | 6.0% |
| 30 | entity_masked_v3 | Complement Naive Bayes | Combined + function words | 24.4% | 15.7% | 24.5% | 34.7% | 93.5% | 6.0% |
| 31 | entity_masked_v3 | Complement Naive Bayes | Chinese function words | 23.6% | 13.8% | 24.5% | 37.5% | 89.7% | 6.0% |
| 32 | entity_masked_v3 | SGD logistic regression | Punctuation/dialogue | 21.5% | 21.9% | 47.2% | 30.5% | 24.8% | 6.0% |
| 33 | entity_masked_v3 | Complement Naive Bayes | Function-character | 19.2% | 8.9% | 15.1% | 41.1% | 86.3% | 6.0% |
| 34 | entity_masked_v3 | Complement Naive Bayes | Combined interpretable | 17.4% | 10.7% | 17.0% | 27.7% | 88.7% | 6.0% |
| 35 | entity_masked_v3 | SGD logistic regression | Sentence/paragraph length | 16.7% | 15.3% | 28.3% | 26.3% | 19.8% | 6.0% |
| 36 | entity_masked_v3 | Complement Naive Bayes | Punctuation/dialogue | 14.0% | 7.9% | 13.2% | 21.0% | 79.6% | 6.0% |
| 37 | entity_masked_v3 | Complement Naive Bayes | Sentence/paragraph length | 7.9% | 4.1% | 9.4% | 15.3% | 73.8% | 6.0% |

## Overall Ranking

| Rank | View | Classifier | Features | Chunk acc. | Balanced acc. | Book acc. | Target F1 | Target recall | Majority baseline |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | entity_masked_v3 | SGD linear SVM (unweighted) | Character n-grams | 88.2% | 84.1% | 94.3% | 85.2% | 100.0% | 6.0% |
| 2 | entity_masked_v3 | SGD linear SVM | Character n-grams | 87.8% | 83.6% | 90.6% | 85.9% | 100.0% | 6.0% |
| 3 | entity_masked_v3 | SGD linear SVM | Character n-grams (hashed) | 85.0% | 80.5% | 92.5% | 77.5% | 100.0% | 6.0% |
| 4 | entity_masked_v3 | SGD linear SVM (unweighted) | Character n-grams (hashed) | 84.0% | 78.7% | 90.6% | 70.7% | 100.0% | 6.0% |
| 5 | entity_masked_v3 | Passive-aggressive linear classifier | Character n-grams (hashed) | 81.7% | 76.4% | 86.8% | 67.8% | 100.0% | 6.0% |
| 6 | clean | SGD linear SVM (unweighted) | Character n-grams | 71.4% | 63.5% | 83.0% | 53.3% | 100.0% | 6.0% |
| 7 | clean | SGD linear SVM | Character n-grams | 70.7% | 63.3% | 77.4% | 56.0% | 100.0% | 6.0% |
| 8 | clean | Passive-aggressive linear classifier | Combined + function words | 69.7% | 64.6% | 90.6% | 80.3% | 98.7% | 6.0% |
| 9 | entity_masked_v3 | Passive-aggressive linear classifier | Combined + function words | 69.5% | 63.9% | 92.5% | 78.7% | 98.7% | 6.0% |
| 10 | entity_masked_v3 | SGD logistic regression | Character n-grams (hashed) | 67.8% | 58.8% | 69.8% | 53.5% | 99.7% | 6.0% |
| 11 | clean | SGD linear SVM | Combined + function words | 67.6% | 63.3% | 90.6% | 86.4% | 95.1% | 6.0% |
| 12 | clean | SGD linear SVM (unweighted) | Combined + function words | 67.3% | 62.7% | 88.7% | 78.8% | 98.8% | 6.0% |
| 13 | entity_masked_v3 | SGD linear SVM | Combined + function words | 67.2% | 62.9% | 92.5% | 79.3% | 96.4% | 6.0% |
| 14 | entity_masked_v3 | SGD linear SVM (unweighted) | Combined + function words | 65.4% | 61.5% | 92.5% | 78.1% | 98.2% | 6.0% |
| 15 | clean | SGD linear SVM | Character n-grams (hashed) | 64.1% | 56.9% | 67.9% | 37.3% | 100.0% | 6.0% |

## Per-Author Weak Spots

Lowest-recall authors under the recommended method:

| Author | Precision | Recall | F1 | Support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 吾九殿 | 100.0% | 5.8% | 10.9% | 468 | 27 | 0 | 441 |
| 青色羽翼 | 97.7% | 15.1% | 26.1% | 837 | 126 | 3 | 711 |
| 酱子贝 | 98.0% | 20.6% | 34.0% | 238 | 49 | 1 | 189 |
| 语笑阑珊 | 63.8% | 29.7% | 40.5% | 475 | 141 | 80 | 334 |
| 多金少女猫 | 99.3% | 32.7% | 49.2% | 459 | 150 | 1 | 309 |
| 龙柒 | 87.5% | 41.8% | 56.6% | 638 | 267 | 38 | 371 |
| 一十四洲 | 75.7% | 47.3% | 58.2% | 368 | 174 | 56 | 194 |
| 红口白牙 | 95.9% | 57.3% | 71.7% | 288 | 165 | 7 | 123 |
| 唐酒卿 | 88.4% | 71.8% | 79.2% | 616 | 442 | 58 | 174 |
| 李温酒 | 99.5% | 74.0% | 84.9% | 1490 | 1102 | 5 | 388 |
| 绿野千鹤 | 90.9% | 75.2% | 82.3% | 266 | 200 | 20 | 66 |
| 北南 | 69.5% | 79.0% | 74.0% | 257 | 203 | 89 | 54 |
| 小霄 | 100.0% | 80.2% | 89.0% | 378 | 303 | 0 | 75 |
| 七流 | 65.4% | 83.9% | 73.6% | 598 | 502 | 265 | 96 |
| 引路星 | 96.6% | 84.0% | 89.8% | 200 | 168 | 6 | 32 |
| 骑鲸南去 | 97.5% | 86.0% | 91.4% | 833 | 716 | 18 | 117 |
| 莫晨欢 | 96.4% | 87.4% | 91.7% | 770 | 673 | 25 | 97 |
| 吕天逸 | 86.0% | 88.3% | 87.1% | 230 | 203 | 33 | 27 |
| 公子于歌 | 97.6% | 90.8% | 94.1% | 457 | 415 | 10 | 42 |
| 微风几许 | 80.3% | 91.2% | 85.4% | 205 | 187 | 46 | 18 |

![Per-author recall](../../generated/style_research/benchmarks/author_classifier_retest_50authors_report/charts/best_method_per_author_recall.svg)

## Top Confusions

| Gold author | Predicted author | Count |
|---|---|---:|
| 吾九殿 | 妄鸦 | 435 |
| 龙柒 | 蝶之灵 | 335 |
| 李温酒 | 蝶之灵 | 227 |
| 青色羽翼 | 非天夜翔 | 183 |
| 语笑阑珊 | 比卡比 | 152 |
| 语笑阑珊 | 非天夜翔 | 123 |
| 青色羽翼 | 七流 | 111 |
| 青色羽翼 | 蝶之灵 | 103 |
| 唐酒卿 | 非天夜翔 | 75 |
| 一十四洲 | 妄鸦 | 72 |

## Interpretation

- The previous concern that the classifier was too weak is addressed for this dataset and split: the best masked chunk-level result is above 80%, and the balanced accuracy is also above 80%.
- N-grams are essential. Interpretable-only features are useful diagnostics but do not reach the target chunk-level accuracy.
- Richer Chinese function-word features materially improve the interpretable baselines: the best combined-rich function-word run reaches 69.5% masked chunk accuracy / 63.9% balanced accuracy, and function words plus function characters reaches 84.3% target-author F1. This is useful as a guardrail, but still below the exact n-gram meter.
- Exact n-grams outperform the hashed approximation in the final hinge run, so the exact model should be treated as the preferred proxy meter when runtime allows.
- The clean-to-masked direction needs caution. Masked > clean suggests the masking process may improve cross-book consistency, but it also means we should keep monitoring whether masks introduce regular artifacts that classifiers exploit.

## Reproduction

```bash
uv run author-style-research authorship-supervised \
  --views clean,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods char_hashing,punctuation_dialogue,length_shape,function_chars,combined_interpretable \
  --classifiers sgd_logistic,complement_nb \
  --max-char-features 262144 \
  --char-min-df 5 \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter1_sgd_nb \
  --jobs -1 --max-iter 3000

uv run author-style-research authorship-supervised \
  --views clean,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods char_hashing \
  --classifiers sgd_logistic,sgd_logistic_unbalanced,sgd_hinge,sgd_hinge_unbalanced,passive_aggressive,complement_nb \
  --max-char-features 262144 \
  --char-min-df 5 \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter2_hash_classifier_sweep \
  --jobs -1 --max-iter 3000

uv run author-style-research authorship-supervised \
  --views clean,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods char_ngrams \
  --classifiers sgd_hinge,sgd_hinge_unbalanced \
  --max-char-features 80000 \
  --char-min-df 20 \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20 \
  --jobs -1 --max-iter 3000

uv run author-style-research authorship-supervised \
  --views clean,entity_masked_v3 \
  --masked-view entity_masked_v3 \
  --methods function_chars,function_words,function_words_plus_chars,combined_interpretable,combined_rich_function_words \
  --classifiers sgd_logistic,sgd_hinge,sgd_hinge_unbalanced,passive_aggressive,complement_nb \
  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter4_function_words \
  --jobs -1 --max-iter 3000

uv run author-style-research authorship-report \
  --evaluation-file generated/style_research/benchmarks/author_classifier_retest_50authors_report/evaluator_review.md \
  --function-word-evaluation-file generated/style_research/benchmarks/author_classifier_retest_50authors_report/function_word_evaluator_review.md
```

## Independent Evaluation

## Evaluation: Eternal Gate Author-Style Classifier Retest

### 1. Accuracy Claim

Yes, the claim is supported, but it should be stated as a masked 50-author supervised classification result, not a general proof of style quality.

The strongest support is:

- `author_style_supervised_50authors_iter3_exact_hinge_mindf20`
- `sgd_hinge_unbalanced.char_ngrams.entity_masked_v3`
- Test accuracy: **88.2%**
- Balanced accuracy: **84.1%**
- Macro F1: **83.5%**
- Book test accuracy: **94.3%**

For target-author use, the class-balanced variant is preferable:

- `sgd_hinge.char_ngrams.entity_masked_v3`
- Test accuracy: **87.8%**
- Balanced accuracy: **83.6%**
- Target F1: **85.9%**
- Target precision/recall: **75.4% / 100.0%**

### 2. Entity Masking

Author signal clearly survives entity masking. Surprisingly, the strongest char-ngram models improve after masking:

- Iter3 balanced exact hinge: clean **70.7% acc / 63.3% balanced acc / 56.0% target F1** vs masked **87.8% / 83.6% / 85.9%**.
- Iter2 hashed hinge: clean **64.1% / 56.9% / 37.3% target F1** vs masked **85.0% / 80.5% / 77.5%**.
- Interpretable features also survive masking: iter1 combined interpretable stays around **50.0% accuracy** and **73.9% masked target F1**.

The surprising direction is that masking helps rather than hurts. That may mean entity names were adding book/topic noise, but it also raises a preprocessing-signal risk: placeholder frequency and masking artifacts may themselves become author-correlated.

### 3. Methodological Caveats

The split is book-level, which is appropriate, but the effective test unit is still only **53 books**, not 35,812 independent samples. Chunk-level metrics should be treated as dense evidence, not independent replication.

Class imbalance is substantial. The target author has **29 train books / 5,650 train chunks**, while most authors have about three books total. This makes `非天夜翔` target recall easier to inflate; the target had **2,150 TP / 0 FN**, but still **703 FP** in the recommended balanced model.

Leakage risk remains. Entity masking removes many names/numbers, but raw text still may contain source boilerplate, genre/setting signals, recurrent placeholders, and topic residues. The masked performance increase should be audited, not simply celebrated.

Hashing vs exact matters. Iter3 exact char n-grams with `min_df=20` outperform iter2 hashed features and are more auditable. Hashing should not be the preferred final meter.

Book aggregation is promising but unstable: most authors contribute one test book, and target book-level support is only four test books. Report book-level results as a sanity check, not the main proof.

### 4. Recommended Meter

Use **class-balanced `sgd_hinge` with exact character n-grams on `entity_masked_v3`, `min_df=20`** as the next proxy style-transfer meter.

For generated-output experiments, report target-author **margin/rank and target chunk share aggregated by chapter/book**, not only hard 50-way accuracy. Keep the iter1 combined-interpretable/function-character models as guardrails: they are weaker but less dependent on high-dimensional lexical n-grams.

### 5. Confusion Checks To Highlight

Highlight false positives into the target author. In the recommended model, major `非天夜翔` false-positive sources include:

- `青色羽翼 -> 非天夜翔`: 183 chunks
- `语笑阑珊 -> 非天夜翔`: 123 chunks
- `唐酒卿 -> 非天夜翔`: 75 chunks

Also highlight persistent non-target confusion clusters:

- `吾九殿 -> 妄鸦`: 435
- `龙柒 -> 蝶之灵`: 335
- `李温酒 -> 蝶之灵`: 227

Per-author low-recall checks should include `吾九殿`, `青色羽翼`, `酱子贝`, `语笑阑珊`, `多金少女猫`, `龙柒`, and `一十四洲`. Low-precision sink labels include `妄鸦`, `蝶之灵`, `比卡比`, and `非天夜翔`.

### Inspected Paths

- `generated/style_research/benchmarks/author_style_supervised_50authors_iter1_sgd_nb/supervised_author_baseline_results.md`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter1_sgd_nb/supervised_author_baseline_results.json`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter2_hash_classifier_sweep/supervised_author_baseline_results.md`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter2_hash_classifier_sweep/supervised_author_baseline_results.json`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/supervised_author_baseline_results.md`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/supervised_author_baseline_results.json`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/supervised_masked_gap.csv`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/confusion_matrices/sgd_hinge.char_ngrams.entity_masked_v3.test.csv`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20/confusion_matrices/sgd_hinge_unbalanced.char_ngrams.entity_masked_v3.test.csv`
- `generated/style_research/corpus/splits.json`
- `generated/style_research/corpus/chunk_report.json`
- `datasets/unmasked/chunks.clean.jsonl`
- `datasets/masked/chunks.entity_masked_v3.jsonl`

No workspace files were modified.

## Independent Evaluation: Function-Word Extension

### What Was Tested

`function_words` is a hand-built Chinese function-phrase lexicon grouped into connective, modal/aspect, deictic/pronoun, particle phrase, and preposition/frame features. It counts exact regex phrase hits, group counts, phrase-density buckets, and diversity buckets.

`function_words_plus_chars` combines those phrase features with the earlier single-character function-character inventory and function-character ratio bucket.

`combined_rich_function_words` adds the richer function-word features to the broader interpretable bundle: punctuation/dialogue, sentence/paragraph length shape, and function characters.

These are credible richer function-word features for an interpretable baseline: they are no longer only single high-frequency characters, and they capture connective/aspect/particle habits that can plausibly reflect style. They are still shallow, regex-based features, not a full Chinese grammatical analysis.

### Best Masked Results

| Scope | Best masked method | Chunk acc. | Balanced acc. | Macro F1 | Target F1 | Target P/R | Book acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Function-word extension, overall | `passive_aggressive` + `combined_rich_function_words` | 69.5% | 63.9% | 60.3% | 78.7% | 65.5% / 98.7% | 92.5% |
| Function-word extension, target F1 | `passive_aggressive` + `function_words_plus_chars` | 60.0% | 53.9% | 50.6% | 84.3% | 79.2% / 90.1% | 90.6% |
| Prior exact n-gram overall | unweighted `sgd_hinge` + exact char n-grams | 88.2% | 84.1% | 83.5% | 85.2% | recall 100.0% | 94.3% |
| Prior target meter | balanced `sgd_hinge` + exact char n-grams | 87.8% | 83.6% | 82.3% | 85.9% | 75.4% / 100.0% | 90.6% |

### Interpretation

The richer function-word methods improve the interpretable side materially, but they do not change the prior conclusion. The best `combined_rich_function_words` run is still about 18.7 points behind exact n-grams in chunk accuracy and about 20.2 points behind in balanced accuracy. The best target-author F1 from `function_words_plus_chars` is close to the exact n-gram target F1, but its overall 50-way discrimination is much weaker.

Exact character n-grams should remain the main style meter for this retest, preferably the class-balanced exact `sgd_hinge` model for target-author scoring.

### Guardrail Use

These methods are useful secondary guardrails because they are lower-dimensional, more interpretable, and less dependent on exact lexical fragments than n-grams. They can catch cases where generated text matches high-dimensional n-gram surface cues but misses broader connective, particle, punctuation, dialogue, or length-shape habits.

The right use is as a corroborating signal: report them beside exact n-gram margin/rank or target share, not as the decisive classifier.

### Caveats

`combined_rich_function_words` is not a pure function-word meter; it includes punctuation/dialogue and length-shape features. `function_words` is regex substring matching with a hand-built lexicon, no segmentation, no syntactic disambiguation, and non-overlapping matches. Some function-word entries are broader phrase markers rather than strict function words.

Interpretation also needs the earlier caveats: chunk-level metrics are dense evidence over a book-level split, not 35,812 independent test samples; book-level target support is small; `非天夜翔` has unusually strong target support; and masked-text performance can still exploit masking artifacts or residual genre/topic signals.

### Inspected Paths

- `workflows/benchmark_author_style.py`
- `workflows/benchmark_author_style_supervised.py`
- `generated/style_research/benchmarks/author_style_supervised_50authors_iter4_function_words/supervised_author_baseline_results.json`
- `docs/reports/02_authorship_style_meter.md`

## Conclusion

Use the exact character n-gram + class-balanced SGD hinge classifier on `entity_masked_v3` as the next target-author proxy style meter. Keep the unweighted exact hinge model as the best overall 50-way classifier, and keep the hashed n-gram model as a faster iteration/debugging approximation. Do not proceed with interpretable-only scoring for style-transfer method selection because it is below the required chunk-level accuracy.
